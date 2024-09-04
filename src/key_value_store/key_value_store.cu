#include "key_value_store.cuh"
#define THREAD_ID (threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x)

std::string getCommandString(CommandType command) {
    static const std::map<CommandType, std::string> commandStrings = {
        {CommandType::PUT, "PLIOPS_Put"},
        {CommandType::GET, "PLIOPS_Get"},
        {CommandType::MULTI_PUT, "PLIOPS_MultiPut"},
        {CommandType::MULTI_GET, "PLIOPS_MultiGet"},
        {CommandType::DELETE, "PLIOPS_Delete"},
        {CommandType::ASYNC_GET_FINALIZE, "ASYNC_GET_FINALIZE"},
        {CommandType::ASYNC_GET_INITIATE, "ASYNC_GET_INITIATE"}
    };

    auto it = commandStrings.find(command);
    if (it != commandStrings.end()) {
        return it->second;
    } else {
        return "Unknown command";
    }
}

__device__ 
void copyData(unsigned char *dst, const unsigned char *src, int value_size, const int tid) {
    int blockSize = blockDim.x;
    for (int i = tid; i < value_size; i += blockSize) {
        dst[i] = src[i];
    }
}

int readEnvVar(const char* varName) {
    const char *env_var = std::getenv(varName);
    if (env_var == NULL) {
        std::cout << "The environment variable " << varName << " is not set.\n";
        std::exit(EXIT_FAILURE);
    }
    return std::atoi(env_var);
}


// KeyHasher

#ifdef IN_MEMORY_STORE
size_t KeyHasher::hash(std::array<unsigned char, MAX_KEY_SIZE> key) const {
    std::hash<unsigned char> hasher;
    size_t hash = 0;
    for (size_t i = 0; i < MAX_KEY_SIZE; ++i) {
        hash ^= hasher(key[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}

bool KeyHasher::equal(const std::array<unsigned char, MAX_KEY_SIZE>& key1, const std::array<unsigned char, MAX_KEY_SIZE>& key2) const {
    return key1 == key2;
}
#endif

// KVMemHandle

KVMemHandle::KVMemHandle()
{
#if !defined(STORELIB_LOOPBACK) && !defined(IN_MEMORY_STORE)
    identify = readEnvVar("DB_IDENTIFY"); 
#endif
}

// RequestMessage

RequestMessage::RequestMessage(int keySize) : keySize(keySize) {
    allocationType = AllocationType::SHARED_CPU_MEMORY;
    CUDA_ERRCHECK(cudaHostAlloc(&key, keySize * sizeof(unsigned char), cudaHostAllocMapped));
}

RequestMessage::~RequestMessage() {
    if(allocationType == AllocationType::SHARED_CPU_MEMORY){
        CUDA_ERRCHECK(cudaFreeHost(key));
    }
    else if(allocationType == AllocationType::CPU_MEMORY){
        delete[] static_cast<unsigned char*>(key);
    }
}

RequestMessage::RequestMessage(const RequestMessage& other) 
    : request_id(other.request_id), 
    cmd(other.cmd), 
    keySize(other.keySize), 
    buffs(other.buffs),
    buffSize(other.buffSize), 
    KVStatus(other.KVStatus),
    numKeys(other.numKeys), 
    ticket(other.ticket) 
    {
    allocationType = AllocationType::CPU_MEMORY;
    key = static_cast<void*>(new unsigned char[keySize]);

    // CUDA_ERRCHECK(cudaHostAlloc(&key, keySize * sizeof(unsigned char), cudaHostAllocMapped));
    for (size_t i = 0; i < keySize; i++)
    {
        ((unsigned char*)key)[i]= ((unsigned char*) other.key)[i];
    }
}

// DataBank

DataBank::DataBank(int queueSize, int maxValueSize, AllocationType allocationType):
    queueSize(queueSize),
    maxValueSize(maxValueSize),
    allocationType(allocationType){
    if(allocationType == AllocationType::SHARED_CPU_MEMORY){
        CUDA_ERRCHECK(cudaHostAlloc((void**)&data, queueSize * maxValueSize * sizeof(unsigned char), cudaHostAllocMapped));
    }
    else if(allocationType == AllocationType::SHARED_GPU_MEMORY){
        cudaGPUMemAlloc(sharedGPUDataBank, queueSize * maxValueSize * sizeof(unsigned char));
        data = (unsigned char*)sharedGPUDataBank.getHostPtr();
    }
}

DataBank::~DataBank() {
    if(allocationType == AllocationType::SHARED_CPU_MEMORY){
        CUDA_ERRCHECK(cudaFreeHost(data));
    }
    else if(allocationType == AllocationType::SHARED_GPU_MEMORY){
        cudaGPUMemFree(sharedGPUDataBank);
    }
}

// ThreadBlockResources

ThreadBlockResources::ThreadBlockResources(int maxNumKeys){
    CUDA_ERRCHECK(cudaMalloc((void**)&StorelibStatus, maxNumKeys * sizeof(int)));
}

ThreadBlockResources::~ThreadBlockResources(){
    CUDA_ERRCHECK(cudaFree(StorelibStatus));
}

// HostAllocatedSubmissionQueue

__device__
bool HostAllocatedSubmissionQueue::getTailAndCheckFull(ThreadBlockResources* d_tbResources, const int tid, int incrementSize){
    BEGIN_THREAD_ZERO {
        d_tbResources->currTail = tail.load(cuda::memory_order_relaxed); // TODO guy CS AFTER         
        d_tbResources->isQueueFull = d_tbResources->currTail - head.load(cuda::memory_order_acquire) + incrementSize - 1 >= this->queueSize;
        d_tbResources->currModTail = d_tbResources->currTail % this->queueSize;
    } END_THREAD_ZERO
    
    return d_tbResources->isQueueFull;
}

__device__
void HostAllocatedSubmissionQueue::incrementTail(ThreadBlockResources* d_tbResources, int incrementSize){
    tail.store((d_tbResources->currTail + incrementSize), cuda::memory_order_release);
}

__device__ 
inline void HostAllocatedSubmissionQueue::setRequestMessage(int idx, CommandType cmd, uint request_id, void* key, int keySize, int numKeys) {
    req_msg_arr[idx].cmd = cmd;
    req_msg_arr[idx].request_id = request_id++;
    for(size_t i = 0; i < keySize; i++)
        ((unsigned char*)(req_msg_arr[idx].key))[i] = ((unsigned char*)key)[i];
    req_msg_arr[idx].keySize = keySize;
    req_msg_arr[idx].numKeys = numKeys;
}

HostAllocatedSubmissionQueue::HostAllocatedSubmissionQueue(gdr_mh_t &mh, int queueSize, int maxKeySize) :
    mh(mh),
    queueSize(queueSize) {
    this->queueStatus.store(false, std::memory_order_relaxed);

    CUDA_ERRCHECK(cudaHostAlloc((void**)&req_msg_arr, queueSize * sizeof(RequestMessage), cudaHostAllocMapped));
    for(int i = 0; i < queueSize; i++){
        new (&req_msg_arr[i]) RequestMessage(maxKeySize);
    }
    head.store(0);
    tail.store(0);
}

HostAllocatedSubmissionQueue::~HostAllocatedSubmissionQueue(){
    for(int i = 0; i < this->queueSize; i++)
        req_msg_arr[i].~RequestMessage();
    CUDA_ERRCHECK(cudaFreeHost(req_msg_arr));
}

__device__
void HostAllocatedSubmissionQueue::copyMetaDataDelete(ThreadBlockResources* d_tbResources, CommandType cmd, const uint request_id, void** keys, int keySize, int numKeys, const int tid){
    int idx;
    int blockSize = blockDim.x;
    for (int i = tid; i < numKeys; i += blockSize) 
    {    
        idx = (d_tbResources->currModTail + i) % this->queueSize;
        setRequestMessage(idx, cmd, request_id, keys[i], keySize, numKeys);
    }
}

__device__
void HostAllocatedSubmissionQueue::copyMetaDataPutAndGet(ThreadBlockResources* d_tbResources, CommandType cmd, const uint request_id, void** keys, int keySize, int buffSize, int numKeys, const int tid){
    int idx;
    int blockSize = blockDim.x;
    for (int i = tid; i < numKeys; i += blockSize) 
    {    
        idx = (d_tbResources->currModTail + i) % this->queueSize;
        setRequestMessage(idx, cmd, request_id, keys[i], keySize, numKeys);
        req_msg_arr[idx].buffSize = buffSize;
    }
}

__device__
void HostAllocatedSubmissionQueue::copyMetaDataAsyncGet(ThreadBlockResources* d_tbResources, CommandType cmd, const uint request_id, void** keys, int keySize, void** buffs, int buffSize, KVStatusType KVStatus[], int numKeys, const int tid){
    int idx;
    int blockSize = blockDim.x;
    // Assigned by all threads in the thread block
    req_msg_arr[d_tbResources->currModTail].buffSize = buffSize;
    req_msg_arr[d_tbResources->currModTail].buffs = buffs;
    req_msg_arr[d_tbResources->currModTail].KVStatus = KVStatus;
    for (int i = tid; i < numKeys; i += blockSize) 
    {    
        idx = (d_tbResources->currModTail + i) % this->queueSize;
        setRequestMessage(idx, cmd, request_id, keys[i], keySize, numKeys);
    }
}

__device__ 
bool HostAllocatedSubmissionQueue::push_put(ThreadBlockResources* d_tbResources, const int tid, DataBank* d_databank_p, CommandType cmd, const uint request_id, void** keys, int keySize, int buffSize, void** buffs, int numKeys) {
    if (getTailAndCheckFull(d_tbResources, tid, numKeys))
        return false;

    int idx;
    for (size_t i = 0; i < numKeys; i++)
    {
        idx = ((d_tbResources->currModTail + i) % this->queueSize) * d_databank_p->maxValueSize;
        copyData((unsigned char*)(&d_databank_p->data[idx]), (const unsigned char*)buffs[i], buffSize, tid);
    }

    copyMetaDataPutAndGet(d_tbResources, cmd, request_id, keys, keySize, buffSize, numKeys, tid);
    BEGIN_THREAD_ZERO { 
        incrementTail(d_tbResources, numKeys);
    } END_THREAD_ZERO

    return true;
}

__device__ 
bool HostAllocatedSubmissionQueue::push_get(ThreadBlockResources* d_tbResources, const int tid, CommandType cmd, const uint request_id, void** keys, int keySize, int buffSize, int numKeys) {
    if (getTailAndCheckFull(d_tbResources, tid, numKeys))
        return false;

    copyMetaDataPutAndGet(d_tbResources, cmd, request_id, keys, keySize, buffSize, numKeys, tid);
    BEGIN_THREAD_ZERO { 
        incrementTail(d_tbResources, numKeys);
    } END_THREAD_ZERO

    return true; 
}

__device__ 
bool HostAllocatedSubmissionQueue::push_async_get_initiate(ThreadBlockResources* d_tbResources, const int tid, CommandType cmd, const uint request_id, void** keys, int keySize, void** buffs, int buffSize, KVStatusType KVStatus[], int numKeys) {
    if (getTailAndCheckFull(d_tbResources, tid, numKeys))
        return false;

    copyMetaDataAsyncGet(d_tbResources, cmd, request_id, keys, keySize, buffs, buffSize, KVStatus, numKeys, tid);
    BEGIN_THREAD_ZERO { 
        incrementTail(d_tbResources, numKeys);
    } END_THREAD_ZERO

    return true; 
}

__device__ 
bool HostAllocatedSubmissionQueue::push_delete(ThreadBlockResources* d_tbResources, const int tid, CommandType cmd, const uint request_id, void** keys, int keySize, int numKeys) {
    if (getTailAndCheckFull(d_tbResources, tid, numKeys))
        return false;

    copyMetaDataDelete(d_tbResources, cmd, request_id, keys, keySize, numKeys, tid); 
    BEGIN_THREAD_ZERO {    
        incrementTail(d_tbResources, numKeys);
    } END_THREAD_ZERO

    return true;
}

__device__ 
bool HostAllocatedSubmissionQueue::push_no_data(ThreadBlockResources* d_tbResources, const int tid, CommandType cmd, const uint request_id, int ticket, int numKeys) {
    if (getTailAndCheckFull(d_tbResources, tid, numKeys))
        return false;
    BEGIN_THREAD_ZERO {
        req_msg_arr[d_tbResources->currModTail].ticket = ticket;
        req_msg_arr[d_tbResources->currModTail].cmd = cmd;
        req_msg_arr[d_tbResources->currModTail].request_id = request_id;
        req_msg_arr[d_tbResources->currModTail].numKeys = numKeys;
        incrementTail(d_tbResources, numKeys);
    } END_THREAD_ZERO

    return true;
}

__host__ 
int HostAllocatedSubmissionQueue::pop(int& currHead) {
    int currModHead = currHead % this->queueSize;
    int incrementSize = req_msg_arr[currModHead].numKeys;
    head.store(currHead + incrementSize, cuda::memory_order_release);
    return currModHead;
}

__host__ 
bool HostAllocatedSubmissionQueue::checkQueueStatus(int &currHead) {
    currHead = head.load(cuda::memory_order_relaxed);
    if (currHead == tail.load(cuda::memory_order_acquire)) {
        return false; // Queue empty
    }
    return true;
}

__host__
void HostAllocatedSubmissionQueue::notifyQueueFull() {
    this->queueStatus.store(true, std::memory_order_release);
    this->queueCondVar.notify_one(); 
}

__host__
void HostAllocatedSubmissionQueue::waitUntilFull(std::unique_lock<std::mutex>& lock) {
    this->queueCondVar.wait(lock, [this] { 
        return this->queueStatus.load(std::memory_order_acquire); 
    });
}

// DeviceAllocatedCompletionQueue    
__device__
void DeviceAllocatedCompletionQueue::copyResponseMessage(ThreadBlockResources* d_tbResources, const int tid, int num_keys, KVStatusType KVStatus[]) {
    int blockSize = blockDim.x;
    ResponseMessage *cq_res_msg = &((ResponseMessage*)sharedGPUDataBankResArr.getDevicePtr())[d_tbResources->currModHead];
    for (int i = tid; i < num_keys; i += blockSize) 
    {
        d_tbResources->StorelibStatus[i] = cq_res_msg->StorelibStatus_d[i];
        KVStatus[i] = cq_res_msg->KVStatus_d[i];
    } 
}

__device__
bool DeviceAllocatedCompletionQueue::getHeadAndCheckEmpty(ThreadBlockResources* d_tbResources, const int tid){
    BEGIN_THREAD_ZERO {
        d_tbResources->currHead = head.load(cuda::memory_order_relaxed); //TODO guy CS AFTER?
        d_tbResources->isQueueEmpty = d_tbResources->currHead == tail.load(cuda::memory_order_acquire);
        d_tbResources->currModHead = d_tbResources->currHead % this->queueSize;
    } END_THREAD_ZERO
    
    return d_tbResources->isQueueEmpty;
}

__device__
void DeviceAllocatedCompletionQueue::incrementHead(ThreadBlockResources* d_tbResources, int incrementSize){
    head.store(d_tbResources->currHead + incrementSize, cuda::memory_order_release);
}

inline void DeviceAllocatedCompletionQueue::setResMsgArrPtrs(size_t maxNumKeys) {
    char* KVStatusIdx = (char*)res_msg_arr + this->queueSize * sizeof(ResponseMessage);
    char* KVStatusIdx_d = (char*)sharedGPUDataBankResArr.getDevicePtr() + this->queueSize * sizeof(ResponseMessage);
    size_t KVStatusOff = maxNumKeys * (sizeof(int) + sizeof(KVStatusType));
    size_t StorelibStatusOff = maxNumKeys * sizeof(KVStatusType);
    for(size_t i = 0; i < this->queueSize; i++) {
        res_msg_arr[i].KVStatus = (KVStatusType*)(KVStatusIdx + KVStatusOff * i);
        res_msg_arr[i].KVStatus_d = (KVStatusType*)(KVStatusIdx_d + KVStatusOff * i);
        res_msg_arr[i].StorelibStatus = (int*)((char*)res_msg_arr[i].KVStatus + StorelibStatusOff);
        res_msg_arr[i].StorelibStatus_d = (int*)((char*)res_msg_arr[i].KVStatus_d + StorelibStatusOff);
    }
}

DeviceAllocatedCompletionQueue::DeviceAllocatedCompletionQueue(gdr_mh_t &mh, int queueSize, int maxNumKeys) :
    mh(mh),
    queueSize(queueSize){
    head.store(0);
    tail.store(0);
    sizeofResMsgArr = queueSize * (sizeof(ResponseMessage) + maxNumKeys * (sizeof(int) + sizeof(KVStatusType)));
    cudaGPUMemAlloc(sharedGPUDataBankResArr, sizeofResMsgArr);
    res_msg_arr = (ResponseMessage*)sharedGPUDataBankResArr.getHostPtr();
    setResMsgArrPtrs(maxNumKeys);
}

DeviceAllocatedCompletionQueue::~DeviceAllocatedCompletionQueue() {
    cudaGPUMemFree(sharedGPUDataBankResArr);
}

__host__
bool DeviceAllocatedCompletionQueue::push(KeyValueStore *kvStore, KVMemHandle &kvMemHandle, int blockIndex, int currModHead, RequestMessage &req_msg) {
    int incrementSize = req_msg.numKeys;
    int currTail = tail.load(cuda::memory_order_relaxed);
    if (currTail - head.load(cuda::memory_order_acquire) + incrementSize - 1 >= this->queueSize) {
        return false; // Queue full
    }
    kvStore->process_kv_request(kvMemHandle, blockIndex, res_msg_arr, currTail, req_msg);
    tail.store(currTail + incrementSize, cuda::memory_order_release);
    return true;
}

__device__ 
bool DeviceAllocatedCompletionQueue::pop_get(ThreadBlockResources* d_tbResources, void* buffs[], int buffSize, const int tid, DataBank* d_databank_p, CommandType cmd, KVStatusType KVStatus[], int numKeys) {
    if (getHeadAndCheckEmpty(d_tbResources, tid))
        return false; 
    unsigned char* d_data_p = (unsigned char*)d_databank_p->sharedGPUDataBank.getDevicePtr();
    int idx;
    for (size_t i = 0; i < numKeys; i++)
    {
        idx = ((d_tbResources->currModTail + i) % this->queueSize) * d_databank_p->maxValueSize;
        copyData((unsigned char*)buffs[i], (const unsigned char*)(&d_data_p[idx]), buffSize, tid);
    }
    copyResponseMessage(d_tbResources, tid, numKeys, KVStatus);
    BEGIN_THREAD_ZERO {
        incrementHead(d_tbResources, numKeys);
    } END_THREAD_ZERO
    return true;
}

__device__ 
bool DeviceAllocatedCompletionQueue::pop_default(ThreadBlockResources* d_tbResources, const int tid, KVStatusType KVStatus[], int numKeys) {
    if (getHeadAndCheckEmpty(d_tbResources, tid))
        return false; 

    copyResponseMessage(d_tbResources, tid, numKeys, KVStatus);
    BEGIN_THREAD_ZERO {
        incrementHead(d_tbResources, numKeys);
    } END_THREAD_ZERO
    return true;
}

__device__ 
bool DeviceAllocatedCompletionQueue::pop_no_res_msg(ThreadBlockResources* d_tbResources, const int tid, int numKeys) {
    if (getHeadAndCheckEmpty(d_tbResources, tid))
        return false; 

    BEGIN_THREAD_ZERO {
        incrementHead(d_tbResources, numKeys);
    } END_THREAD_ZERO
    return true;
}

__device__ 
bool DeviceAllocatedCompletionQueue::pop_async_get_init(ThreadBlockResources* d_tbResources, const int tid, unsigned int *p_ticket, int numKeys) {
    if (getHeadAndCheckEmpty(d_tbResources, tid))
        return false; 

    *p_ticket = d_tbResources->currTail;

    BEGIN_THREAD_ZERO {
        incrementHead(d_tbResources, numKeys);
    } END_THREAD_ZERO
    return true;
}

// DeviceCompletionQueueWithDataBank

DeviceCompletionQueueWithDataBank::DeviceCompletionQueueWithDataBank(gdr_mh_t &mh, int queueSize, int maxValueSize, int maxNumKeys) :
        cq(mh, queueSize, maxNumKeys),
        dataBank(queueSize, maxValueSize, AllocationType::SHARED_GPU_MEMORY){}

// HostSubmissionQueueWithDataBank

HostSubmissionQueueWithDataBank::HostSubmissionQueueWithDataBank(gdr_mh_t &mh, int queueSize, int maxValueSize, int maxKeySize):
        sq(mh, queueSize, maxKeySize),
        dataBank(queueSize, maxValueSize, AllocationType::SHARED_CPU_MEMORY){}


void handle_status(KVStatusType &KVStatus, int StorelibStatus, CommandType cmd, uint request_id, void *key){
    if (StorelibStatus == 5) // Key does not exist
    {
        KVStatus = KVStatusType::NON_EXIST;
    }
    else if (StorelibStatus != 0) { // Any error othen than key does not exist
        printf("%s Failed, ret=%d\n", getCommandString(cmd).c_str(), StorelibStatus);
        printf("request_id = %d, key = %d\n", request_id, *((int*)key));
        printf("---------------------------------------------------\n");
        KVStatus = KVStatusType::FAIL;
    }
    else
        KVStatus = KVStatusType::SUCCESS;
}

#ifdef IN_MEMORY_STORE
void putInMemoryStore(RequestMessage &req_msg, void* data, tbb::concurrent_hash_map<std::array<unsigned char, MAX_KEY_SIZE>, std::array<unsigned char, MAX_VALUE_SIZE>, KeyHasher> &inMemoryStoreMap, KVStatusType &res_ans) {
    std::array<unsigned char, MAX_KEY_SIZE> keyArray = {};
    std::copy(static_cast<unsigned char*>(req_msg.key), 
        static_cast<unsigned char*>(req_msg.key) + req_msg.keySize, 
        keyArray.begin());

    std::array<unsigned char, MAX_VALUE_SIZE> valArray;
    std::copy(static_cast<unsigned char*>(data), 
          static_cast<unsigned char*>(data) + req_msg.buffSize, 
          valArray.begin());
    inMemoryStoreMap.insert({keyArray, valArray});

    res_ans= KVStatusType::SUCCESS;
}

void getFromMemoryStore(RequestMessage &req_msg, void* data, tbb::concurrent_hash_map<std::array<unsigned char, MAX_KEY_SIZE>, std::array<unsigned char, MAX_VALUE_SIZE>, KeyHasher> &inMemoryStoreMap, KVStatusType &res_ans, void* key) {
    tbb::concurrent_hash_map<std::array<unsigned char, MAX_KEY_SIZE>, std::array<unsigned char, MAX_VALUE_SIZE>, KeyHasher>::accessor a;
    std::array<unsigned char, MAX_KEY_SIZE> keyArray = {};
    std::copy((unsigned char*)key, (unsigned char*)key + req_msg.keySize, keyArray.begin());
    if (inMemoryStoreMap.find(a, keyArray)) {
        std::array<unsigned char, MAX_VALUE_SIZE> &val = a->second;
        std::copy(val.begin(), val.begin() + req_msg.buffSize, static_cast<unsigned char*>(data));
    }
    else {
        // TODO guy Key not found handling 
    }
    res_ans= KVStatusType::SUCCESS;
}

#else
void putInPliopsDB(PLIOPS_DB_t &plio_handle, RequestMessage &req_msg, void *data, KVStatusType &KVStatus, int &StorelibStatus) {
    StorelibStatus = PLIOPS_Put(plio_handle, req_msg.key, req_msg.keySize, data, req_msg.buffSize, NO_OPTIONS);
    handle_status(KVStatus, StorelibStatus, req_msg.cmd, req_msg.request_id, req_msg.key);
}

void getFromPliopsDB(PLIOPS_DB_t &plio_handle, RequestMessage &req_msg, void *data, KVStatusType &KVStatus, int &StorelibStatus, void* key) {
    uint get_actual_object_size;
    StorelibStatus = PLIOPS_Get(plio_handle, key, req_msg.keySize, data, req_msg.buffSize, &get_actual_object_size);
    handle_status(KVStatus, StorelibStatus, req_msg.cmd, req_msg.request_id, key);
}
#endif

// KeyValueStore
__device__ 
void KeyValueStore::KVPutBaseD(void* keys[], const unsigned int keySize, void* buffs[], const unsigned int buffSize, KVStatusType KVStatus[], CommandType cmd, int numKeys) {
    HostSubmissionQueueWithDataBank *d_hostmem_p = this->h_hostmem_p;
    DeviceCompletionQueueWithDataBank *d_devmem_p = (DeviceCompletionQueueWithDataBank *)this->sharedGPUCompletionQueueWithDataBank.getDevicePtr();

    int blockIndex = blockIdx.x;
    const int tid = THREAD_ID;
                    
    ThreadBlockResources &tbResources = this->d_tbResources[blockIndex];
    HostAllocatedSubmissionQueue *submission_queue = &d_hostmem_p[blockIndex].sq;
    DeviceAllocatedCompletionQueue *completion_queue = &d_devmem_p[blockIndex].cq;
    DataBank* d_hostDataBank_p = &d_hostmem_p[blockIndex].dataBank;
    
    // TODO handle later: check if completion queue is not full as well
    while (!submission_queue->push_put(&tbResources, tid, d_hostDataBank_p, cmd, tbResources.request_id, keys, keySize, buffSize, buffs, numKeys));
    // Immediately wait for a response
    while (!completion_queue->pop_default(&tbResources, tid, KVStatus, numKeys));
}

__device__ 
void KeyValueStore::KVGetBaseD(void* keys[], const unsigned int keySize, void* buffs[], const unsigned int buffSize, KVStatusType KVStatus[], CommandType cmd, int numKeys) {
    HostSubmissionQueueWithDataBank *d_hostmem_p = this->h_hostmem_p;
    DeviceCompletionQueueWithDataBank *d_devmem_p = (DeviceCompletionQueueWithDataBank *)this->sharedGPUCompletionQueueWithDataBank.getDevicePtr();
    int blockIndex = blockIdx.x;
    const int tid = THREAD_ID;
    ThreadBlockResources &tbResources = this->d_tbResources[blockIndex];
    HostAllocatedSubmissionQueue *submission_queue = &d_hostmem_p[blockIndex].sq;
    DeviceAllocatedCompletionQueue *completion_queue = &d_devmem_p[blockIndex].cq;
    DataBank* d_devDataBank_p = &d_devmem_p[blockIndex].dataBank;
    while (!submission_queue->push_get(&tbResources, tid, cmd, tbResources.request_id, keys, keySize, buffSize, numKeys)); // Busy-wait until the value is pushed successfully
    // Immediately wait for a response
    while (!completion_queue->pop_get(&tbResources, buffs, buffSize, tid, d_devDataBank_p, cmd, KVStatus, numKeys));
}

void KeyValueStore::server_func(KVMemHandle &kvMemHandle, int blockIndex) {
    HostAllocatedSubmissionQueue *submission_queue = &h_hostmem_p[blockIndex].sq;
    DeviceAllocatedCompletionQueue *completion_queue = &h_devmem_p[blockIndex].cq;

    CommandType command = CommandType::NONE;
    
    while (command != CommandType::EXIT) {

        // wait until the queue is full
        std::unique_lock<std::mutex> lock(submission_queue->queueMutex);
        submission_queue->waitUntilFull(lock);

        int currModHead = submission_queue->pop(submission_queue->currHead);
        // Set the first new request message
        RequestMessage &req_msg = submission_queue->req_msg_arr[currModHead];
        // Parse the request message
        command = submission_queue->req_msg_arr[currModHead].cmd;
        while (!completion_queue->push(this, kvMemHandle, blockIndex, currModHead, req_msg)); // Busy-wait until the value is pushed successfully
        submission_queue->queueStatus.store(false, std::memory_order_release);
    }
}

void KeyValueStore::process_async_get(KVMemHandle &kvMemHandle, int blockIndex, ResponseMessage &res_msg, int currModTail, size_t num_keys, void* keys_buffer, RequestMessage* p_req_msg_cpy) {
#ifndef IN_MEMORY_STORE // XDP
    PLIOPS_DB_t& plio_handle = kvMemHandle.plio_handle;
    tbb::parallel_for(size_t(0), num_keys, [&](size_t i) {
        getFromPliopsDB(plio_handle, 
        *p_req_msg_cpy,
        p_req_msg_cpy->buffs[i],
        p_req_msg_cpy->KVStatus[i],
        res_msg.StorelibStatus[i],
        (char*)keys_buffer + i * p_req_msg_cpy->keySize);
    });
#else // IN_MEMORY_STORE
    tbb::concurrent_hash_map<std::array<unsigned char, MAX_KEY_SIZE>, std::array<unsigned char, MAX_VALUE_SIZE>, KeyHasher>& inMemoryStoreMap = kvMemHandle.inMemoryStoreMap;
    tbb::parallel_for(size_t(0), num_keys, [&](size_t i) {
        getFromMemoryStore(*p_req_msg_cpy, 
        p_req_msg_cpy->buffs[i],
        inMemoryStoreMap, 
        res_msg.KVStatus[i],
        (char*)keys_buffer + i * p_req_msg_cpy->keySize);
    });
#endif
    delete[] static_cast<unsigned char*>(keys_buffer);
    delete p_req_msg_cpy;
}

void printMapContents(const tbb::concurrent_hash_map<int, std::shared_future<void>>& ticketToFutureMap) { // TODO guy DELETE
    for (tbb::concurrent_hash_map<int, std::shared_future<void>>::const_iterator it = ticketToFutureMap.begin(); it != ticketToFutureMap.end(); ++it) {
        int key = it->first;
        // Since std::shared_future<void> does not have a direct way to print its value,
        // we can only print the key and indicate that there is a future associated with it.
        std::cout << "Key: " << key << ", Value: [shared_future<void>]" << std::endl;
    }
}

void KeyValueStore::process_kv_request(KVMemHandle &kvMemHandle, int blockIndex, ResponseMessage *res_msg_arr, int currTail, RequestMessage &req_msg){
    int currModTail = currTail % queueSize;
    ResponseMessage res_msg = res_msg_arr[currModTail];
    #ifndef STORELIB_LOOPBACK
    DataBank* h_hostDataBank_p = &(this->h_hostmem_p[blockIndex].dataBank);
    DataBank* h_devDataBank_p = &(this->h_devmem_p[blockIndex].dataBank);
    HostAllocatedSubmissionQueue *submission_queue = &this->h_hostmem_p[blockIndex].sq;
    int &queueSize = submission_queue->queueSize;

#endif
    size_t num_keys = req_msg.numKeys;

#ifdef STORELIB_LOOPBACK
#elif defined(IN_MEMORY_STORE)
    tbb::concurrent_hash_map<std::array<unsigned char, MAX_KEY_SIZE>, std::array<unsigned char, MAX_VALUE_SIZE>, KeyHasher>& inMemoryStoreMap = kvMemHandle.inMemoryStoreMap;
#else
    PLIOPS_DB_t& plio_handle = kvMemHandle.plio_handle;
    int ret = 0;
#endif
    CommandType command = req_msg.cmd;
    if (command == CommandType::EXIT){
        res_msg.KVStatus[0] = KVStatusType::EXIT;
    }
    else if (command == CommandType::PUT || command == CommandType::MULTI_PUT)
    {
#ifdef STORELIB_LOOPBACK
        for (int i = 0; i < num_keys; i++)
            res_msg.KVStatus[i] = KVStatusType::SUCCESS;

#elif defined(IN_MEMORY_STORE)
        tbb::parallel_for(size_t(0), num_keys, [&](size_t i) {
            putInMemoryStore(submission_queue->req_msg_arr[(currModTail + i) % queueSize],
            &h_hostDataBank_p->data[(currModTail + i) % queueSize * h_hostDataBank_p->maxValueSize], 
            inMemoryStoreMap, 
            res_msg.KVStatus[i]);
        });
#else
        tbb::parallel_for(size_t(0), num_keys, [&](size_t i) {
            putInPliopsDB(plio_handle, submission_queue->req_msg_arr[(currModTail + i) % queueSize], &h_hostDataBank_p->data[(currModTail + i) % queueSize * h_hostDataBank_p->maxValueSize], res_msg.KVStatus[i], res_msg.StorelibStatus[i]);
        });
        
#endif
    }
    else if (command == CommandType::GET || command == CommandType::MULTI_GET)
    {
#ifdef STORELIB_LOOPBACK
        for (int i = 0; i < num_keys; i++)
            res_msg.KVStatus[i] = KVStatusType::SUCCESS;
#elif defined(IN_MEMORY_STORE)
        tbb::parallel_for(size_t(0), num_keys, [&](size_t i) {
            getFromMemoryStore(submission_queue->req_msg_arr[(currModTail + i) % queueSize], 
            &h_devDataBank_p->data[(currModTail + i) % queueSize * h_devDataBank_p->maxValueSize], 
            inMemoryStoreMap, 
            res_msg.KVStatus[i], 
            submission_queue->req_msg_arr[(currModTail + i) % queueSize].key);
        });
#else
        tbb::parallel_for(size_t(0), num_keys, [&](size_t i) {
            getFromPliopsDB(plio_handle, 
            submission_queue->req_msg_arr[(currModTail + i) % queueSize], 
            &h_devDataBank_p->data[(currModTail + i) % queueSize * h_devDataBank_p->maxValueSize], 
            res_msg.KVStatus[i], 
            res_msg.StorelibStatus[i], 
            submission_queue->req_msg_arr[(currModTail + i) % queueSize].key);
        });
#endif
    }
    else if (command == CommandType::ASYNC_GET_INITIATE){
        void* keys_buffer = new unsigned char[num_keys * req_msg.keySize];
        for (size_t i = 0; i < num_keys; i++)
            for(size_t j = 0; j < req_msg.keySize; j++)
                ((unsigned char*)keys_buffer)[i * req_msg.keySize + j] = ((unsigned char*)(submission_queue->req_msg_arr[(currModTail + i) % submission_queue->queueSize].key))[j];

        RequestMessage* p_req_msg_cpy = new RequestMessage(req_msg);
        std::shared_future<void> asyncResult = std::async(std::launch::async, 
        &KeyValueStore::process_async_get, 
        this, 
        std::ref(kvMemHandle), 
        blockIndex, 
        std::ref(res_msg), 
        currModTail, 
        num_keys, 
        keys_buffer, 
        p_req_msg_cpy);
        ticketToFutureMapArr[blockIndex].insert({currTail, asyncResult}); // TODO guy switch to currHead? think about it
    }
    else if (command == CommandType::ASYNC_GET_FINALIZE){
        int ticket = submission_queue->req_msg_arr[currModTail].ticket;
        tbb::concurrent_hash_map<int, std::shared_future<void>>::accessor a;
        std::shared_future<void> asyncResult;
        if (ticketToFutureMapArr[blockIndex].find(a, ticket)) {
            asyncResult = a->second;
            asyncResult.wait();
            // Finally, erase the ticket from the map
            ticketToFutureMapArr[blockIndex].erase(a);
        }
        else {
            printf("KEY NOT FOUND!, ticket = %d\n", ticket); // TODO guy handle this better
        }
        
    }
    else if (command == CommandType::DELETE)
    {
#ifdef STORELIB_LOOPBACK
        res_msg.KVStatus[0] = KVStatusType::SUCCESS;
#elif defined(IN_MEMORY_STORE)
        res_msg.KVStatus[0] = KVStatusType::SUCCESS;
#else
        ret = PLIOPS_Delete(plio_handle, &req_msg.key, req_msg.keySize, NO_OPTIONS);
        handle_status(res_msg.KVStatus[0], ret,  req_msg.cmd, req_msg.request_id, req_msg.key);
#endif
    }
    else
    {
        //std::cout << "Cannot perform command " << (int)req_msg.cmd << std::endl;
        for (int i = 0; i < num_keys; i++)
            res_msg.KVStatus[i] = KVStatusType::FAIL;
    }  
}

bool KeyValueStore::checkParameters(int queueSize, int maxValueSize, int maxNumKeys, int maxKeySize) {
    if(maxNumKeys < 1){
        std::cerr << "Error: the num keys must be greater than 0." << std::endl;
        return false;
    }
    if(queueSize < maxNumKeys){
        std::cerr << "Error: the queue size must be equal to or greater than num_keys." << std::endl;
        return false;
    }
    if(maxValueSize < 1){
        std::cerr << "Error: the value size must be greater than 0." << std::endl;
        return false;
    }
    if(maxKeySize < 1){
        std::cerr << "Error: the key size must be greater than 0." << std::endl;
        return false;
    }
    return true;
}

void KeyValueStore::checkIfQueuesAreFull(int numThreadBlocks) {
    HostAllocatedSubmissionQueue *submission_queue;
    while(!this->isExit) {
        for (size_t blockIndex = 0; blockIndex < numThreadBlocks; ++blockIndex) {
            submission_queue = &h_hostmem_p[blockIndex].sq;
            std::unique_lock<std::mutex> lock(submission_queue->queueMutex, std::defer_lock);
            if(!submission_queue->queueStatus.load(std::memory_order_acquire) && lock.try_lock()) {
                if(submission_queue->checkQueueStatus(submission_queue->currHead)) {
                    // Notify the thread that the queue is full
                    submission_queue->notifyQueueFull();
                }
            }
        }
    }
}

KeyValueStore::KeyValueStore(const int numThreadBlocks, const int blockSize, int maxValueSize, int maxNumKeys, int maxKeySize, KVMemHandle &kvMemHandle) {            
#ifdef IN_MEMORY_STORE
    maxValueSize = MAX_VALUE_SIZE;
    maxKeySize = MAX_KEY_SIZE;
    printf("Using in-memory store preconfigured values: maxValueSize = %d, maxKeySize = %d\n", maxValueSize, maxKeySize);
#endif
    
    this->numThreadBlocks = numThreadBlocks;
    this->blockSize = blockSize;
    this->pKVMemHandle = &kvMemHandle;

    YAML::Node config = YAML::LoadFile(CONFIG_YAML_PATH);
    queueSize = config["RUNTIME"]["KV_STORE"]["QUEUE_SIZE"].as<int>();

    if(!checkParameters(queueSize, maxValueSize, maxNumKeys, maxKeySize))
        throw std::runtime_error("Error: Invalid parameters");

    ticketToFutureMapArr = new tbb::concurrent_hash_map<int, std::shared_future<void>>[numThreadBlocks];
    
    // Allocate memory that is shared by the CPU and the GPU
    CUDA_ERRCHECK(cudaHostAlloc((void **)&h_hostmem_p, numThreadBlocks * sizeof(HostSubmissionQueueWithDataBank), cudaHostAllocMapped));
    cudaGPUMemAlloc(sharedGPUCompletionQueueWithDataBank, numThreadBlocks * sizeof(DeviceCompletionQueueWithDataBank));
    h_devmem_p = (DeviceCompletionQueueWithDataBank*)sharedGPUCompletionQueueWithDataBank.getHostPtr();

    for (size_t i = 0; i < numThreadBlocks; i++) {
        new (&h_hostmem_p[i]) HostSubmissionQueueWithDataBank(sharedGPUCompletionQueueWithDataBank.mh, queueSize, maxValueSize, maxKeySize);
        new (&h_devmem_p[i]) DeviceCompletionQueueWithDataBank(sharedGPUCompletionQueueWithDataBank.mh, queueSize, maxValueSize, maxNumKeys);
    }

    ThreadBlockResources* h_tbResources = (ThreadBlockResources*)malloc(numThreadBlocks * sizeof(ThreadBlockResources));
    for(size_t i = 0; i < numThreadBlocks; ++i)
        new (&h_tbResources[i]) ThreadBlockResources(maxNumKeys);
        
    CUDA_ERRCHECK(cudaMalloc(&d_tbResources, numThreadBlocks * sizeof(ThreadBlockResources)));
    CUDA_ERRCHECK(cudaMemcpy(d_tbResources, h_tbResources, numThreadBlocks * sizeof(ThreadBlockResources), cudaMemcpyHostToDevice)); 
    free(h_tbResources);

    // Open server threads - in the near future this will be done internally in the library and will not be invoked manually by the user
    std::vector<std::thread> threads; // To keep track of the threads, in case you want to join them later
    for (int blockIndex = 0; blockIndex < numThreadBlocks; ++blockIndex) {
        // Launch a thread with a different index
        std::thread(&KeyValueStore::server_func, this, std::ref(kvMemHandle), blockIndex).detach();
    }
    
    // for (int blockIndex = 0; blockIndex < numThreadBlocks; ++blockIndex) {
    //     threads.emplace_back([&, blockIndex]() {
    //         // Create a cpu_set_t object representing a set of CPUs. Clear it and set the CPU you want the thread to run on.
    //         cpu_set_t cpuset;
    //         CPU_ZERO(&cpuset);
    //         CPU_SET(blockIndex, &cpuset); // Set the CPU to the current index (i)
    //         // Apply the CPU set to the current thread (the one created by emplace_back)
    //         int thread_policy = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    //         if (thread_policy != 0) {
    //             std::cerr << "Error setting thread affinity: " << std::strerror(thread_policy) << std::endl;
    //         }
    //         // Call the server function
    //         server_func(kvStore, plio_handle, blockIndex);
    //     });
    // }

    for (auto& thread : threads) {
        thread.detach();
    }

    // Launch a thread that checks if there is a full queue, and wakes up the appropriate thread
    std::thread(&KeyValueStore::checkIfQueuesAreFull, this, numThreadBlocks).detach();
}

KeyValueStore::~KeyValueStore() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    KVExitD<<<numThreadBlocks, blockSize, 0, stream>>>(this);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    this->isExit = true;
    
    ThreadBlockResources* h_tbResources = (ThreadBlockResources*)malloc(numThreadBlocks * sizeof(ThreadBlockResources));
    CUDA_ERRCHECK(cudaMemcpy(h_tbResources, d_tbResources, numThreadBlocks * sizeof(ThreadBlockResources), cudaMemcpyDeviceToHost));
    for(size_t i = 0; i < numThreadBlocks; ++i) {
        h_tbResources[i].~ThreadBlockResources();
    }
    CUDA_ERRCHECK(cudaFree(d_tbResources));
    free(h_tbResources);

    for (size_t i = 0; i < numThreadBlocks; i++) {
        h_hostmem_p[i].~HostSubmissionQueueWithDataBank();
        h_devmem_p[i].~DeviceCompletionQueueWithDataBank();
    }
    
    // Deallocate memory that is shared by the CPU and the GPU
    cudaGPUMemFree(sharedGPUCompletionQueueWithDataBank);
    CUDA_ERRCHECK(cudaFreeHost(h_hostmem_p));

    delete[] ticketToFutureMapArr;
}

bool KeyValueStore::KVOpenDB(KVMemHandle& kvMemHandle) {
#if !defined(STORELIB_LOOPBACK) && !defined(IN_MEMORY_STORE)
    PLIOPS_IDENTIFY_t& identify = kvMemHandle.identify;
    PLIOPS_DB_t& plio_handle= kvMemHandle.plio_handle;
    PLIOPS_DB_OPEN_OPTIONS_t db_open_options;
    db_open_options.createIfMissing = 1;
    db_open_options.tailSizeInBytes = 0;
    db_open_options.errorIfExists = false;

    int ret = PLIOPS_OpenDB(identify, &db_open_options, XDP_ID, &plio_handle);
    if (ret != 0) {
        printf("PLIOPS_OpenDB Failed ret=%d\n", ret);
        return false;
    }
    std::cout << "Finished PLIOPS_OpenDB!" << std::endl;   
#endif
    return true;
}

bool KeyValueStore::KVCloseDB(KVMemHandle& kvMemHandle) {
#if !defined(STORELIB_LOOPBACK) && !defined(IN_MEMORY_STORE)
    PLIOPS_DB_t& plio_handle= kvMemHandle.plio_handle;
    int ret = PLIOPS_CloseDB(plio_handle);
    if (ret != 0) {
        printf("PLIOPS_CloseDB Failed ret=%d\n", ret);
        return false;
    }
    std::cout << "Finished PLIOPS_CloseDB!" << std::endl;       
#endif
    return true;
}

bool KeyValueStore::KVDeleteDB(KVMemHandle& kvMemHandle) {
#ifndef XDP_ON_HOST
#if !defined(STORELIB_LOOPBACK) && !defined(IN_MEMORY_STORE)
    PLIOPS_IDENTIFY_t& identify = kvMemHandle.identify;
    int ret = PLIOPS_DeleteDB(identify, NO_OPTIONS);
    if (ret != 0) {
        printf("PLIOPS_DeleteDB Failed ret=%d\n", ret);
        return false;
    }
    std::cout << "Finished PLIOPS_DeleteDB!" <<std::endl;  
#endif
    return true;
#else
    std::cout << "KVDeleteDB is not supported in XDP on host" <<std::endl;  
    return false;
#endif
}


__device__ 
void KeyValueStore::KVPutD(void* key, unsigned int keySize, void* buff, unsigned int buffSize, KVStatusType &KVStatus) {
    void* buffs[] = {buff};
    void* keys[] = {key};
    KVStatusType KVStatuses[] = {KVStatus};
    KVPutBaseD(keys, keySize, buffs, buffSize, KVStatuses, CommandType::PUT);
}

__device__ 
void KeyValueStore::KVMultiPutD(void* keys[], unsigned int keySize, void* buffs[], unsigned int buffSize, KVStatusType KVStatus[], int numKeys) {
    KVPutBaseD(keys, keySize, buffs, buffSize, KVStatus, CommandType::MULTI_PUT, numKeys);
}

__device__ 
void KeyValueStore::KVGetD(void* key, const unsigned int keySize, void* buff, const unsigned int buffSize, KVStatusType &KVStatus) {         
    void* buffs[] = {buff};
    void* keys[] = {key};
    KVStatusType KVStatuses[] = {KVStatus};
    KVGetBaseD(keys, keySize, buffs, buffSize, KVStatuses, CommandType::GET);
}

__device__ 
void KeyValueStore::KVMultiGetD(void* keys[], const unsigned int keySize, void* buffs[], const unsigned int buffSize, KVStatusType KVStatus[], int numKeys) {
    KVGetBaseD(keys, keySize, buffs, buffSize, KVStatus, CommandType::MULTI_GET, numKeys);
    }

__device__ 
void KeyValueStore::KVDeleteD(void* key, unsigned int keySize, KVStatusType KVStatus[]) {
    HostSubmissionQueueWithDataBank *d_hostmem_p = this->h_hostmem_p;
    DeviceCompletionQueueWithDataBank *d_devmem_p = (DeviceCompletionQueueWithDataBank *)this->sharedGPUCompletionQueueWithDataBank.getDevicePtr();

    int blockIndex = blockIdx.x;
    const int tid = THREAD_ID;
                    
    ThreadBlockResources &tbResources = this->d_tbResources[blockIndex];
    HostAllocatedSubmissionQueue *submission_queue = &d_hostmem_p[blockIndex].sq;
    DeviceAllocatedCompletionQueue *completion_queue = &d_devmem_p[blockIndex].cq;
    void* keys[] = {key};

    // TODO handle later: check if completion queue is not full as well
    while (!submission_queue->push_delete(&tbResources, tid, CommandType::DELETE, tbResources.request_id, keys, keySize));
    // Immediately wait for a response
    while (!completion_queue->pop_default(&tbResources, tid, KVStatus));
}

__device__ 
void KeyValueStore::KVAsyncGetInitiateD(void* keys[], const unsigned int keySize, GPUMultiBufferHandle& valMultiBuff, const unsigned int buffSize, GPUMultiBufferHandle& kvStatusMultiBuff, int numKeys, unsigned int *p_ticket) {
    HostSubmissionQueueWithDataBank *d_hostmem_p = h_hostmem_p;
    DeviceCompletionQueueWithDataBank *d_devmem_p = (DeviceCompletionQueueWithDataBank *)sharedGPUCompletionQueueWithDataBank.getDevicePtr();

    int blockIndex = blockIdx.x;
    const int tid = THREAD_ID;
                    
    ThreadBlockResources &tbResources = d_tbResources[blockIndex];
    HostAllocatedSubmissionQueue *submission_queue = &d_hostmem_p[blockIndex].sq;
    DeviceAllocatedCompletionQueue *completion_queue = &d_devmem_p[blockIndex].cq;
    
    void** buffs = (void**)valMultiBuff.getHostPtr();
    KVStatusType* KVStatus = (KVStatusType*)kvStatusMultiBuff.getHostPtr();
    while (!submission_queue->push_async_get_initiate(&tbResources, tid, CommandType::ASYNC_GET_INITIATE, tbResources.request_id, keys, keySize, buffs, buffSize, KVStatus, numKeys));
    // Immediately wait for a response
    while (!completion_queue->pop_async_get_init(&tbResources, tid, p_ticket, numKeys));
}

__device__
void KeyValueStore::KVAsyncGetFinalizeD(unsigned int ticket){
    HostSubmissionQueueWithDataBank *d_hostmem_p = h_hostmem_p;
    DeviceCompletionQueueWithDataBank *d_devmem_p = (DeviceCompletionQueueWithDataBank *)sharedGPUCompletionQueueWithDataBank.getDevicePtr();

    int blockIndex = blockIdx.x;
    const int tid = threadIdx.z * blockDim.y * blockDim.x 
                + threadIdx.y * blockDim.x 
                + threadIdx.x;

    ThreadBlockResources &tbResources = d_tbResources[blockIndex];
    HostAllocatedSubmissionQueue *submission_queue = &d_hostmem_p[blockIndex].sq;
    DeviceAllocatedCompletionQueue *completion_queue = &d_devmem_p[blockIndex].cq;

    while (!submission_queue->push_no_data(&tbResources, tid, CommandType::ASYNC_GET_FINALIZE, tbResources.request_id, ticket));
    // Immediately wait for a response
    while (!completion_queue->pop_no_res_msg(&tbResources, tid));
}

__host__
void KeyValueStore::KVMultiPutH(void* keys[], unsigned int keySize, void* buffs[], unsigned int buffSize, KVStatusType KVStatus[], size_t numKeys){
#ifndef STORELIB_LOOPBACK
#ifndef IN_MEMORY_STORE // XDP
    int *StorelibStatus = (int*)malloc(numKeys * sizeof(int));
    PLIOPS_DB_t plio_handle = pKVMemHandle->plio_handle; 
    tbb::parallel_for(size_t(0), numKeys, [&](size_t i) {
        StorelibStatus[i] = PLIOPS_Put(plio_handle, keys[i], keySize, buffs[i], buffSize, NO_OPTIONS);

        if (StorelibStatus[i] != 0) {
            printf("KVMultiPutH Failed, ret=%d\n", StorelibStatus[i]);
            KVStatus[i] = KVStatusType::FAIL;
        }
        else
            KVStatus[i] = KVStatusType::SUCCESS;

    });

    free(StorelibStatus);

#else // IN_MEMORY_STORE
    tbb::concurrent_hash_map<std::array<unsigned char, MAX_KEY_SIZE>, std::array<unsigned char, MAX_VALUE_SIZE>, KeyHasher> &inMemoryStoreMap = pKVMemHandle->inMemoryStoreMap;
    tbb::parallel_for(size_t(0), numKeys, [&](size_t i) {
        std::array<unsigned char, MAX_KEY_SIZE> keyArray;
        std::copy((unsigned char*)keys[i], (unsigned char*)keys[i] + keySize, keyArray.begin());

        std::array<unsigned char, MAX_VALUE_SIZE> valArray;
        std::copy((unsigned char*)buffs[i], (unsigned char*)buffs[i] + buffSize, valArray.begin());

        if (inMemoryStoreMap.insert({keyArray, valArray}))
            KVStatus[i] = KVStatusType::SUCCESS;
        else
            KVStatus[i] = KVStatusType::FAIL;
    });
#endif
#endif
}

__host__
void KeyValueStore::KVPutH(void* key, unsigned int keySize, void* buff, unsigned int buffSize, KVStatusType &KVStatus){
#ifndef STORELIB_LOOPBACK
#ifndef IN_MEMORY_STORE // XDP
    PLIOPS_STATUS_et StorelibStatus;
    PLIOPS_DB_t plio_handle = pKVMemHandle->plio_handle;
    
    StorelibStatus = PLIOPS_Put(plio_handle, key, keySize, buff, buffSize, NO_OPTIONS);

    if (StorelibStatus != PLIOPS_STATUS_OK) {
        printf("KVPutH Failed, ret=%d\n", StorelibStatus);
        KVStatus = KVStatusType::FAIL;
    }
    else
        KVStatus = KVStatusType::SUCCESS;
#else // IN_MEMORY_STORE
    tbb::concurrent_hash_map<std::array<unsigned char, MAX_KEY_SIZE>, std::array<unsigned char, MAX_VALUE_SIZE>, KeyHasher> &inMemoryStoreMap = pKVMemHandle->inMemoryStoreMap;
    std::array<unsigned char, MAX_KEY_SIZE> keyArray;
    std::copy((unsigned char*)key, (unsigned char*)key + keySize, keyArray.begin());

    std::array<unsigned char, MAX_VALUE_SIZE> valArray;
    std::copy((unsigned char*)buff, (unsigned char*)buff + buffSize, valArray.begin());
    if (inMemoryStoreMap.insert({keyArray, valArray}))
        KVStatus = KVStatusType::SUCCESS;
    else
        KVStatus = KVStatusType::FAIL;
#endif
#endif
}

__global__ 
void KVExitD(KeyValueStore *kvStore) {
    HostSubmissionQueueWithDataBank *d_hostmem_p = kvStore->h_hostmem_p;
    DeviceCompletionQueueWithDataBank *d_devmem_p = (DeviceCompletionQueueWithDataBank *)kvStore->sharedGPUCompletionQueueWithDataBank.getDevicePtr();

    int blockIndex = blockIdx.x;
    const int tid = THREAD_ID;
                    
    ThreadBlockResources &tbResources = kvStore->d_tbResources[blockIndex];
    HostAllocatedSubmissionQueue *submission_queue = &d_hostmem_p[blockIndex].sq;
    DeviceAllocatedCompletionQueue *completion_queue = &d_devmem_p[blockIndex].cq;
    
    while (!submission_queue->push_no_data(&tbResources, tid, CommandType::EXIT, tbResources.request_id));
    // Immediately wait for a response
    while (!completion_queue->pop_no_res_msg(&tbResources, tid));
}