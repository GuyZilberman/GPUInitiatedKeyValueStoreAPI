#ifndef KEY_VALUE_STORE_H
#define KEY_VALUE_STORE_H
#include <atomic>
#include <iostream>
#include <vector>
#include <cuda/atomic>
#include <future>
#include <yaml-cpp/yaml.h>
#include "gdrapi.h"
#include "gdr_gpu_memalloc.cuh"

#if !defined(STORELIB_LOOPBACK) && !defined(IN_MEMORY_STORE)
#ifdef XDP_ON_HOST
#include "/etc/opt/pliops/xdp-onhost/store_lib_expo.h"
#else
#include "/etc/pliops/store_lib_expo.h"
#endif
#endif

#include "tbb/concurrent_hash_map.h"
#include "tbb/parallel_for.h"
#include "common.cuh"

#define XDP_ID 0
#define NO_OPTIONS 0

#include <stdio.h>
#include <time.h>
#include <chrono>
using namespace std::chrono;

#define BEGIN_THREAD_ZERO __syncthreads(); if (tid == 0)
#define END_THREAD_ZERO __syncthreads();

// Forward declaration of the classes
class KeyValueStore;

// External function declaration
__global__ 
void KVExitD(KeyValueStore *kvStore);

int readEnvVar(const char* varName);

#ifdef IN_MEMORY_STORE
struct KeyHasher{
    size_t hash(const std::array<unsigned char, MAX_KEY_SIZE> key) const;

    bool equal(const std::array<unsigned char, MAX_KEY_SIZE>& key1, const std::array<unsigned char, MAX_KEY_SIZE>& key2) const;
};

struct InMemoryValue{
    std::array<unsigned char, MAX_VALUE_SIZE> value;
    int valueSize;
};
#endif

struct KVMemHandle{
#ifdef IN_MEMORY_STORE
    tbb::concurrent_hash_map<std::array<unsigned char, MAX_KEY_SIZE>, InMemoryValue, KeyHasher> inMemoryStoreMap;
#else
    PLIOPS_IDENTIFY_t identify;
    PLIOPS_DB_t plio_handle;
#endif

    KVMemHandle();
};

enum class CommandType {
    NONE,
    EXIT,
    PUT,
    GET,
    DELETE,
    MULTI_PUT,
    MULTI_GET,
    ASYNC_GET_INITIATE,
    ASYNC_GET_FINALIZE,
    ASYNC_PUT,
    ASYNC_GET
};

enum class KVStatusType {
    SUCCESS,
    FAIL,
    NON_EXIST,
    EXIT,
    NONE
};

enum class AllocationType {
    CPU_MEMORY,
    SHARED_GPU_MEMORY,
    SHARED_CPU_MEMORY
};

struct RequestMessage {
	AllocationType allocationType;
    uint request_id;
	CommandType cmd;
	void *key;
    int keySize;
    void** buffs;
    int buffSize;
    KVStatusType* KVStatus;
    int numKeys;
    unsigned int ticket;

    RequestMessage(int maxNumKeys, int maxKeySize);
    RequestMessage(int keySize);
    RequestMessage(const RequestMessage& other);
    ~RequestMessage();
};

struct ResponseMessage {
	KVStatusType* KVStatus;
    KVStatusType* KVStatus_d;
    int* StorelibStatus;
    int* StorelibStatus_d;
};

struct DataBank {
    unsigned char* data;
    int queueSize;
    int maxValueSize;
    AllocationType allocationType;
    GPUBufferHandle sharedGPUDataBank; 

    DataBank(int queueSize, int maxValueSize, AllocationType allocationType);
    ~DataBank();
};

struct ThreadBlockResources {
    int *StorelibStatus;
    uint request_id = 0;
    int currTail = -1;
    int currHead = -1;
    int currModTail = -1;
    int currModHead = -1;
    bool isQueueEmpty = false;
    bool isQueueFull = false;

    ThreadBlockResources(int maxNumKeys);
    ~ThreadBlockResources();
};


class HostAllocatedSubmissionQueue {
private:
    cuda::atomic<int> tail;
    gdr_mh_t mh;

    __device__
    bool getTailAndCheckFull(ThreadBlockResources* d_tbResources, const int tid, int incrementSize);

    __device__
    void incrementTail(ThreadBlockResources* d_tbResources, int incrementSize);

    __device__ 
    inline void setRequestMessage(int idx, CommandType cmd, uint request_id, void* key, int keySize, int numKeys);

public:
    cuda::atomic<int> head;
    RequestMessage* req_msg_arr;
    int queueSize;
    std::atomic<bool> isServerThreadActive;
    std::mutex queueMutex;
    std::condition_variable queueCondVar;
    int currHead;

    HostAllocatedSubmissionQueue(gdr_mh_t &mh, int queueSize, int maxKeySize);

    ~HostAllocatedSubmissionQueue();

    __device__
    void copyMetaDataDeleteAndGet(ThreadBlockResources* d_tbResources, CommandType cmd, const uint request_id, void** keys, int keySize, int numKeys, const int tid);

    __device__
    void copyMetaDataPut(ThreadBlockResources* d_tbResources, CommandType cmd, const uint request_id, void** keys, int keySize, int buffSize, int numKeys, const int tid);

    __device__
    void copyMetaDataAsyncGet(ThreadBlockResources* d_tbResources, CommandType cmd, const uint request_id, void** keys, int keySize, void** buffs, int buffSize, KVStatusType KVStatus[], int numKeys, const int tid);

    __device__ 
    bool push_put(ThreadBlockResources* d_tbResources, const int tid, DataBank* d_databank_p, CommandType cmd, const uint request_id, void** keys, int keySize, int buffSize, void** buffs, int numKeys);
    
    __device__ 
    bool push_get(ThreadBlockResources* d_tbResources, const int tid, CommandType cmd, const uint request_id, void** keys, int keySize, int numKeys);

    __device__
    bool push_async_get_initiate(ThreadBlockResources* d_tbResources, const int tid, CommandType cmd, const uint request_id, void** keys, int keySize, void** buffs, int buffSize, KVStatusType KVStatus[], int numKeys);

    __device__ 
    bool push_delete(ThreadBlockResources* d_tbResources, const int tid, CommandType cmd, const uint request_id, void** keys, int keySize, int numKeys = 1);

    __device__ 
    bool push_no_data(ThreadBlockResources* d_tbResources, const int tid, CommandType cmd, const uint request_id, int ticket = 0, int numKeys = 1);
    
    __host__ 
    void pop(const int currHead, KVMemHandle &kvMemHandle, RequestMessage *async_req_msg, ResponseMessage *async_res_msg, DataBank *dataBank, int &currModHead, CommandType &command, int &numKeys);

    __host__ 
    bool checkQueueNotEmpty(int &currHead);

    __host__
    void controllerSignalServerThread();

    __host__
    void serverThreadWaitUntilSignal(std::unique_lock<std::mutex>& lock);
};


class DeviceAllocatedCompletionQueue {
private:
    cuda::atomic<int> head;
    int sizeofResMsgArr;
    ResponseMessage* res_msg_arr;
    GPUBufferHandle sharedGPUDataBankResArr;
    gdr_mh_t mh;

    __device__
    void copyResponseMessage(ThreadBlockResources* d_tbResources, const int tid, int num_keys, KVStatusType KVStatus[]);

    __device__
    bool getHeadAndCheckEmpty(ThreadBlockResources* d_tbResources, const int tid);
    
    __device__
    void incrementHead(ThreadBlockResources* d_tbResources, int incrementSize);

    inline void setResMsgArrPtrs(size_t maxNumKeys);

public:
    cuda::atomic<int> tail;
    int queueSize;

    DeviceAllocatedCompletionQueue(gdr_mh_t &mh, int queueSize, int maxNumKeys);

    ~DeviceAllocatedCompletionQueue();

    __host__
    bool push(KeyValueStore *kvStore, KVMemHandle &kvMemHandle, RequestMessage *async_req_msg, ResponseMessage *async_res_msg, int blockIndex, int currModHead, CommandType command, const int numKeys);

    __device__ 
    bool pop_get(ThreadBlockResources* d_tbResources, void* buffs[], int buffSize, const int tid, DataBank* d_databank_p, KVStatusType KVStatus[], int numKeys);

    __device__ 
    bool pop_default(ThreadBlockResources* d_tbResources, const int tid, KVStatusType KVStatus[], int numKeys = 1);

    __device__ 
    bool pop_no_res_msg(ThreadBlockResources* d_tbResources, const int tid, int numKeys = 1);

    __device__ 
    bool pop_async_get_init(ThreadBlockResources* d_tbResources, const int tid, unsigned int *p_ticket, int numKeys = 1);};

struct DeviceCompletionQueueWithDataBank {
    DeviceAllocatedCompletionQueue cq;
    DataBank dataBank;

    DeviceCompletionQueueWithDataBank(gdr_mh_t &mh, int queueSize, int maxValueSize, int maxNumKeys);
};

struct HostSubmissionQueueWithDataBank {
    HostAllocatedSubmissionQueue sq;
    DataBank dataBank;

    HostSubmissionQueueWithDataBank(gdr_mh_t &mh, int queueSize, int maxValueSize, int maxKeySize);
};

#ifdef IN_MEMORY_STORE
void putInMemoryStore(RequestMessage &req_msg, void* data, tbb::concurrent_hash_map<std::array<unsigned char, MAX_KEY_SIZE>, InMemoryValue, KeyHasher> &inMemoryStoreMap, KVStatusType &res_ans);
void getFromMemoryStore(RequestMessage &req_msg, void* data, tbb::concurrent_hash_map<std::array<unsigned char, MAX_KEY_SIZE>, InMemoryValue, KeyHasher> &inMemoryStoreMap, KVStatusType &res_ans, void* key);
#else
void putInPliopsDB(PLIOPS_DB_t &plio_handle, RequestMessage &req_msg, void *data, KVStatusType &KVStatus, int &StorelibStatus);
void getFromPliopsDB(PLIOPS_DB_t &plio_handle, RequestMessage &req_msg, void *data, KVStatusType &KVStatus, int &StorelibStatus, void* key, int buffSize);
#endif

class KeyValueStore {
    private:
        static KVMemHandle kvMemHandle;
        KVMemHandle *pKVMemHandle;
        int queueSize;
        tbb::concurrent_hash_map<int, std::shared_future<void>>* ticketToFutureMapArr;
        bool isExit = false;

        friend class DeviceAllocatedCompletionQueue;
        __device__ 
        void KVPutBaseD(void* keys[], unsigned int keySize, void* buffs[], unsigned int buffSize, KVStatusType KVStatus[], CommandType cmd, int numKeys = 1);
    
        __device__ 
        void KVGetBaseD(void* keys[], const unsigned int keySize, void* buffs[], const unsigned int buffSize, KVStatusType KVStatus[], CommandType cmd, int numKeys = 1);

        void server_func(KVMemHandle &kvMemHandle, int blockIndex, int maxNumKeys, int maxKeySize);

        void process_async_get(KVMemHandle &kvMemHandle, int blockIndex, ResponseMessage &res_msg, int currModTail, size_t num_keys, void* keys_buffer, RequestMessage* p_req_msg_cpy);

        void process_kv_request(KVMemHandle &kvMemHandle, int blockIndex, ResponseMessage *curr_res_msg_arr, int currTail, RequestMessage *async_req_msg, ResponseMessage *async_res_msg, CommandType command, const int numKeys);

        bool checkParameters(int queueSize, int maxValueSize, int maxNumKeys, int maxKeySize);

        void controller_func(int numThreadBlocks);
        
    public:
        HostSubmissionQueueWithDataBank* h_hostmem_p;
        DeviceCompletionQueueWithDataBank* h_devmem_p;
        GPUBufferHandle sharedGPUCompletionQueueWithDataBank;
        ThreadBlockResources* d_tbResources;
        int numThreadBlocks;
        int blockSize;

        KeyValueStore(const int numThreadBlocks, const int blockSize, int maxValueSize, int maxNumKeys, int maxKeySize);        
        
        ~KeyValueStore();

        static bool KVOpenDB();

        static bool KVCloseDB();

        static bool KVDeleteDB();

        __device__ 
        void KVPutD(void* key, unsigned int keySize, void* buff, unsigned int buffSize, KVStatusType &KVStatus);

        __device__ 
        void KVMultiPutD(void* keys[], unsigned int keySize, void* buffs[], unsigned int buffSize, KVStatusType KVStatus[], int numKeys);

        __device__ 
        void KVGetD(void* key, const unsigned int keySize, void* buff, const unsigned int buffSize, KVStatusType &KVStatus);

        __device__ 
        void KVMultiGetD(void* keys[], const unsigned int keySize, void* buffs[], const unsigned int buffSize, KVStatusType KVStatus[], int numKeys);

        __device__ 
        void KVDeleteD(void* key, unsigned int keySize, KVStatusType KVStatus[]);
        
        // Async Put
        __device__ 
        void KVAsyncMultiPutInitiateD(void* keys[], unsigned int keySize, void* buffs[], unsigned int buffSize, int numKeys);

        __device__
        void KVAsyncMultiPutFinalizeD(KVStatusType KVStatus[], int numKeys);

        // Async Get
        __device__
        void KVAsyncMultiGetInitiateD(void* keys[], const unsigned int keySize, int numKeys);

        __device__
        void KVAsyncMultiGetFinalizeD(void* buffs[], const unsigned int buffSize, KVStatusType KVStatus[], int numKeys);

        // Async Get ZC
        __device__
        void KVAsyncMultiGetZCInitiateD(void* keys[], const unsigned int keySize, GPUMultiBufferHandle& valMultiBuff, const unsigned int buffSize, GPUMultiBufferHandle& kvStatusMultiBuff, int numKeys, unsigned int *p_ticket);

        __device__
        void KVAsyncMultiGetZCFinalizeD(unsigned int ticket);

        __host__
        void KVMultiPutH(void* keys[], unsigned int keySize, void* buffs[], unsigned int buffSize, KVStatusType KVStatus[], size_t numKeys);
        
        __host__
        void KVPutH(void* key, unsigned int keySize, void* buff, unsigned int buffSize, KVStatusType &KVStatus);
};

KVMemHandle KeyValueStore::kvMemHandle;

class KeyValueStoreVLLM {
public:
    KeyValueStoreVLLM(const int numThreadBlocks, const int blockSize, int maxValueSize, int maxNumKeys, int maxKeySize);

    ~KeyValueStoreVLLM();

    static bool KVOpenDB();

    static bool KVCloseDB();

    static bool KVDeleteDB();

    __device__ 
    void KVPutD(void* key, unsigned int keySize, void* buff, unsigned int buffSize, KVStatusType &KVStatus);

    __device__ 
    void KVMultiPutD(void* keys[], unsigned int keySize, void* buffs[], unsigned int buffSize, KVStatusType KVStatus[], int numKeys);

    __device__ 
    void KVGetD(void* key, const unsigned int keySize, void* buff, const unsigned int buffSize, KVStatusType &KVStatus);

    __device__ 
    void KVMultiGetD(void* keys[], const unsigned int keySize, void* buffs[], const unsigned int buffSize, KVStatusType KVStatus[], int numKeys);

    __device__ 
    void KVDeleteD(void* key, unsigned int keySize, KVStatusType KVStatus[]);
    
    // Async Put
    __device__ 
    void KVAsyncPutInitiateD(void* keys[], unsigned int keySize, void* buffs[], unsigned int buffSize, int numKeys);

    __device__
    void KVAsyncPutFinalizeD(KVStatusType KVStatus[], int numKeys);

    // Async Get
    __device__
    void KVAsyncGetInitiateD(void* keys[], const unsigned int keySize, int numKeys);

    __device__
    void KVAsyncGetFinalizeD(void* buffs[], const unsigned int buffSize, KVStatusType KVStatus[], int numKeys);

    // Async Get ZC
    __device__
    void KVAsyncGetZCInitiateD(void* keys[], const unsigned int keySize, GPUMultiBufferHandle& valMultiBuff, const unsigned int buffSize, GPUMultiBufferHandle& kvStatusMultiBuff, int numKeys, unsigned int *p_ticket);

    __device__
    void KVAsyncGetZCFinalizeD(unsigned int ticket);

    __host__
    void KVMultiPutH(void* keys[], unsigned int keySize, void* buffs[], unsigned int buffSize, KVStatusType KVStatus[], size_t numKeys);
    
    __host__
    void KVPutH(void* key, unsigned int keySize, void* buff, unsigned int buffSize, KVStatusType &KVStatus);

private:
    KeyValueStore *kvStoreR;
    KeyValueStore *kvStoreW;
};

#endif // KEY_VALUE_STORE_H