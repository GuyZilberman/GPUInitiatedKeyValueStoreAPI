# KeyValueStore API: High-Performance Key-Value Store for GPU-Accelerated Systems

## Introduction:

The KeyValueStore API is a high-performance key-value store designed for GPU-accelerated systems. It provides efficient storage operations by leveraging CUDA and GPU-based parallelism, making it ideal for large-scale data processing and compute-intensive applications. This repository contains the full implementation of the KeyValueStore, offering a flexible interface for managing key-value pairs in both CPU and GPU memory. The API supports both in-memory and database-backed stores, enabling seamless integration into a variety of real-time data processing workflows.

## Dependencies

Before getting started, make sure you have the following dependencies installed:

- `install_dependencies.sh`: Run the following command to install and configure all required dependencies:
    ```
    source scripts/install_dependencies.sh
    ```

- `gdrcopy`: You can find detailed installation instructions on the [gdrcopy GitHub repository](https://github.com/NVIDIA/gdrcopy).


## Configuration

To configure the KeyValueStore API, you can modify `cfg/config.yaml`. This file contains various settings that control the behavior of the API. Here are some key configurations you can modify:

### Compile-Time Settings (`COMPILE_TIME`)
These settings are used during the build process of the API.

- **`KV_STORE.IN_MEMORY_STORE`**:
    - `MAX_VALUE_SIZE`: Defines the maximum size for a value stored in the in-memory key-value store. 
      - Example: `4096` (bytes)
    - `MAX_KEY_SIZE`: Defines the maximum size for a key stored in the in-memory key-value store.
      - Example: `4` (bytes)

### Runtime Settings (`RUNTIME`)
These settings are used during the runtime of the API and can be modified without recompiling the code.

- **`KV_STORE.QUEUE_SIZE`**:
    - Defines the size of the queue used for managing incoming requests.
    - Example: `550`

- **`KV_STORE.XDP.DB_IDENTIFY`**:
    - Placeholder for XDP's database identifier. 


## Build

To build the GPU KV API, you can use the following commands:

For the default configuration (only available with Pliops' XDP or XDP on-host):
```
bear make -j
```

For the in-memory store configuration:
```
bear make -j IN_MEMORY_STORE
```
Note: The delete operation is currently unsupported in this mode. This functionality will be added in a future update.


## Usage

Allocate pinned memory accessible by both CPU and GPU for a KeyValueStore instance.
Construct a KeyValueStore object in the allocated memory with specified thread blocks and block size.
```cpp
KVMemHandle kvMemHandle;
KeyValueStore *kvStore;
CUDA_ERRCHECK(cudaHostAlloc((void **)&kvStore, sizeof(KeyValueStore), cudaHostAllocMapped));
new (kvStore) KeyValueStore(numThreadBlocks, blockSize, DATA_ARR_SIZE*sizeof(int), NUM_KEYS, sizeof(int), kvMemHandle);
```

Open the kvStore database with the memory handle, enabling subsequent put and get calls.
```cpp
kvStore->KVOpenDB(kvMemHandle);
```

Launch a CUDA kernel. Note numThreadBlocks must be consistent with the one you passed to the KeyValueStore constructor.
```cpp
myKernel<<<numThreadBlocks, numThreadsPerBlock>>>(kvStore, /*additional kernel args*/); 
```

Use any of the KV API functions, for example:
```cpp
kvStore->KVMultiPutD(keys, sizeof(int), buffs, sizeof(int) * DATA_ARR_SIZE, KVStatus, NUM_KEYS);
kvStore->KVMultiGetD(keys, sizeof(int), buffs, sizeof(int) * DATA_ARR_SIZE, KVStatus, NUM_KEYS);
```

Finally, call KVCloseDB
```cpp
kvStore->KVCloseDB(kvMemHandle);
```

You may delete the DB using KVDeleteDB (Not yet supported on XDP on-host)
```cpp
kvStore->KVDeleteDB(kvMemHandle);
```


## Asynchronous API
### Memory Allocation API
The asynchronous API is ideal for scenarios where overlapping data transfers with computations can provide significant performance improvements.
To use the Asynchronous API, you need to allocate buffers in the following manner:
```cpp
GPUMultiBufferHandle arrOfUserMultiBuffer[CONCURRENT_COUNT]; 
GPUMultiBufferHandle arrOfUserKVStatusArr[CONCURRENT_COUNT];
size_t buffer_size_in_bytes = DATA_ARR_SIZE * sizeof(int);
size_t num_buffers = NUM_KEYS;

for (size_t i = 0; i < CONCURRENT_COUNT; i++)
{
    cudaGPUMultiBufferAlloc(arrOfUserMultiBuffer[i], num_buffers, buffer_size_in_bytes);
    cudaGPUMultiBufferAlloc(arrOfUserKVStatusArr[i], num_buffers, sizeof(KVStatusType));
}
```

Then, you may use the following API to start an async non-blocking multi-get operation
```cpp
kvStore->KVAsyncGetInitiateD(keys, sizeof(int), arrOfUserMultiBuffer[idx], sizeof(int) * DATA_ARR_SIZE, arrOfUserKVStatusArr[idx], NUM_KEYS, &ticket_arr[idx]);
```
And then use the following blocking API to finalize the operation
```cpp
kvStore->KVAsyncGetFinalizeD(ticket_arr[idx]);
```

## Acknowledgement
The codebase builds on top of the open-source gdrcopy project by NVIDIA, available [here](https://github.com/NVIDIA/gdrcopy). We use gdrcopy to enable and extend its efficient shared memory allocation API, which plays a critical role in enabling high-performance data transfers between CPU and GPU memory.