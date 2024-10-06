# GPU Initiated KeyValueStore API: High-Performance Key-Value Store for GPU-Accelerated Systems

## Introduction:

The GPU Initiated KeyValueStore API is a high-performance key-value store designed for GPU-accelerated systems. It provides efficient storage operations by leveraging CUDA and GPU-based parallelism, making it ideal for large-scale data processing and compute-intensive applications. This repository contains the full implementation of the KeyValueStore, offering a flexible interface for managing key-value pairs in both CPU and GPU memory. The API supports both in-memory and database-backed stores, enabling seamless integration into a variety of real-time data processing workflows.

## Dependencies

Before getting started, make sure you have the following dependencies installed:

- `install_dependencies.sh`: Run the following command to install and configure all required dependencies:
    ```
    source scripts/install_dependencies.sh
    ```

- `gdrcopy`: You can find detailed installation instructions on the [gdrcopy GitHub repository](https://github.com/NVIDIA/gdrcopy).


## Installing XDP on Host
### Install
To install XDP on host, follow the instructions below:

Ubuntu 22.04:
```
sudo apt-get install packages/ubuntu22.04/pliops-xdp-onhost_3.0.2.0.deb
```

Ubuntu 20.04:
```
sudo apt-get install packages/ubuntu20.04/pliops-xdp-onhost_3.0.2.0.deb
```

RHEL 9.4:
```
sudo yum install packages/redhat9.4/pliops-xdp-onhost_3.0.2.0.rpm
```
### Configuring XDP on Host:

Once installed, configure the system using the following command, replacing `MEDIA_DEVICES_LIST` with the list of NVMe SSD devices:
```
sudo /etc/opt/pliops/xdp-onhost/xdp-oh_configurator setup --resources="MEDIA_DEVICES_LIST"
```

e.g.
```
sudo /etc/opt/pliops/xdp-onhost/xdp-oh_configurator setup --resources="/dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1"
```
### Uninstalling XDP on Host:

If you need to uninstall XDP on host, first stop XDP on host:
```
sudo /etc/opt/pliops/xdp-onhost/xdp-oh_configurator stop
```

Then follow the appropriate step for your system:

Ubuntu:
```
sudo apt-get remove pliops-xdp-onhost
```

RHEL:
```
sudo yum remove pliops-xdp-onhost
```

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


## Linking Your Application with the KeyValueStore Library
To link your application with the `libkey_value_store.a` library, you can reuse the same compiler, flags, and libraries that are already defined in this project's `Makefile`. Here's how you can add the linking instructions to your own `Makefile`:
```
user_app: $(USER_APP_OBJECTS) libkey_value_store.a
	$(CC) $(CFLAGS) $(USER_APP_CFLAGS) $(USER_APP_OBJECTS) -o $@ $(LFLAGS) $(USER_APP_LFLAGS) $(KV_APP_LIBS)
```

Replace the following placeholders with your specific values:
- `user_app`: Replace this with your application target name.
- `$(USER_APP_OBJECTS)`: The variable that lists all object files for your application.
- `$(USER_APP_CFLAGS)`: Your application's compilation flags.
- `$(USER_APP_LFLAGS)`: Your application's linker flags.
- `$(CC)`, `$(CFLAGS)`, `$(LFLAGS)`, and `$(KV_APP_LIBS)`: These variables are already defined in the KeyValueStore project's `Makefile`, so you don't need to redefine them. Simply include them to maintain consistency with the project's build process.

## Usage

Allocate pinned memory accessible by both CPU and GPU for a KeyValueStore instance.
Construct a KeyValueStore object in the allocated memory with specified thread blocks and block size.
```cpp
KeyValueStore *kvStore;
cudaHostAlloc((void **)&kvStore, sizeof(KeyValueStore), cudaHostAllocMapped);
new (kvStore) KeyValueStore(numThreadBlocks, blockSize, DATA_ARR_SIZE*sizeof(int), NUM_KEYS, sizeof(int));
```

Open the kvStore database with the memory handle, enabling subsequent put and get calls.
```cpp
KeyValueStore::KVOpenDB();
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
KeyValueStore::KVCloseDB();
```

You may delete the DB using KVDeleteDB (Not yet supported on XDP on-host)
```cpp
KeyValueStore::KVDeleteDB();
```


## Asynchronous API
The asynchronous API is ideal for scenarios where overlapping data transfers with computations can provide significant performance improvements.
### Zero-Copy Memory Allocation API
If you wish to use the zero-copy asynchronous API for non-blocking operations, allocate buffers as follows:
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

Then initiate an async non-blocking multi-get operation:
```cpp
kvStore->KVAsyncMultiGetZCInitiateD(keys, sizeof(int), arrOfUserMultiBuffer[idx], sizeof(int) * DATA_ARR_SIZE, arrOfUserKVStatusArr[idx], NUM_KEYS, &ticket_arr[idx]);
```
To finalize the operation:
```cpp
kvStore->KVAsyncMultiGetZCFinalizeD(ticket_arr[idx]);
```

### Simplified Asynchronous API
For scenarios where simplicity is preferred, you can now use a straightforward asynchronous API for put and get operations without needing special memory allocations.

#### Asynchronous Put API
To initiate a non-blocking simple async put operation:
```cpp
kvStore->KVAsyncMultiPutInitiateD((void**)userResources.keys, sizeof(int), (void**)userResources.buffs, sizeof(int) * DATA_ARR_SIZE, NUM_KEYS);
```
To finalize the operation:
```cpp
kvStore->KVAsyncMultiPutFinalizeD(userResources.KVStatus, NUM_KEYS);
```

#### Asynchronous Get API
To initiate a non-blocking simple async get operation:
```cpp
kvStore->KVAsyncMultiGetInitiateD((void**)userResources.keys, sizeof(int), NUM_KEYS);
```
To finalize the operation:
```cpp
kvStore->KVAsyncMultiGetFinalizeD((void**)userResources.buffs, sizeof(int) * DATA_ARR_SIZE, userResources.KVStatus, NUM_KEYS);
```

## Acknowledgement
The codebase builds on top of the open-source gdrcopy project by NVIDIA, available [here](https://github.com/NVIDIA/gdrcopy). We use gdrcopy to enable and extend its efficient shared memory allocation API, which plays a critical role in enabling high-performance data transfers between CPU and GPU memory.