#include <iomanip>
#include <iostream>
using namespace std;

#include "gdrapi.h"
#include "gdrcopy_common.hpp"

using namespace gdrcopy::test;

class GPUDeviceInitializer{
    private:
        void cuda_select_device(int dev_id = 0){
            int n_devices = 0;
            ASSERTDRV(cuDeviceGetCount(&n_devices));

            CUdevice dev;
            for (int n=0; n<n_devices; ++n) {
                
                char dev_name[256];
                int dev_pci_domain_id;
                int dev_pci_bus_id;
                int dev_pci_device_id;

                ASSERTDRV(cuDeviceGet(&dev, n));
                ASSERTDRV(cuDeviceGetName(dev_name, sizeof(dev_name) / sizeof(dev_name[0]), dev));
                ASSERTDRV(cuDeviceGetAttribute(&dev_pci_domain_id, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev));
                ASSERTDRV(cuDeviceGetAttribute(&dev_pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev));
                ASSERTDRV(cuDeviceGetAttribute(&dev_pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev));

                cout  << "GPU id:" << n << "; name: " << dev_name 
                    << "; Bus id: "
                    << std::hex 
                    << std::setfill('0') << std::setw(4) << dev_pci_domain_id
                    << ":" << std::setfill('0') << std::setw(2) << dev_pci_bus_id
                    << ":" << std::setfill('0') << std::setw(2) << dev_pci_device_id
                    << std::dec
                    << endl;
            }
            cout << "selecting device " << dev_id << endl;
            ASSERTDRV(cuDeviceGet(&dev, dev_id));

            CUcontext dev_ctx;
            ASSERTDRV(cuDevicePrimaryCtxRetain(&dev_ctx, dev));
            ASSERTDRV(cuCtxSetCurrent(dev_ctx));

            // Check that the device supports GDR
            ASSERT_EQ(check_gdr_support(dev), true);
        }

    public:
        gdr_t gdrHandle;

        GPUDeviceInitializer () : gdrHandle() {
            // Open a handle to the GPUDirect RDMA driver
            gdrHandle = gdr_open_safe();

            // Initialize the CUDA driver API
            ASSERTDRV(cuInit(0));

            cuda_select_device();
        }
        ~GPUDeviceInitializer (){
            cout << "closing gdrdrv" << endl;
            ASSERT_EQ(gdr_close(gdrHandle), 0);
        }
};

GPUDeviceInitializer gpuDI;

class GPUBufferHandle{
    private:
    public:
        void *host_ptr;
        CUdeviceptr device_ptr; 
        gpu_mem_handle_t mhandle;
        void *mappedDevicePtr;
        int allocationSize;
        gdr_mh_t mh;        

        GPUBufferHandle() : mh(), mhandle(), mappedDevicePtr(nullptr) {}
        
        __device__ __host__
        void* getHostPtr(){
            return host_ptr;
        }

        __device__ __host__
        CUdeviceptr getDevicePtr(){
            return device_ptr;
        }
};

void cudaGPUMemAlloc(GPUBufferHandle& gpuBufHandle, const int allocationSize){
    gdr_t &gdrHandle = gpuDI.gdrHandle;
    gpuBufHandle.allocationSize = allocationSize;

    ASSERTDRV(gpu_mem_alloc(&gpuBufHandle.mhandle, gpuBufHandle.allocationSize, true, true));
    // Final device ptr
    gpuBufHandle.device_ptr = gpuBufHandle.mhandle.ptr;

    BEGIN_CHECK {
        // Create a peer-to-peer mapping of the device memory buffer, returning an opaque handle.
        ASSERT_EQ(gdr_pin_buffer(gdrHandle, gpuBufHandle.device_ptr, gpuBufHandle.allocationSize, 0, 0, &gpuBufHandle.mh), 0);
        ASSERT_NEQ(gpuBufHandle.mh, null_mh);

        // Create a user-space mapping of the memory handle.
        ASSERT_EQ(gdr_map(gdrHandle, gpuBufHandle.mh, &gpuBufHandle.mappedDevicePtr, gpuBufHandle.allocationSize), 0);

        gdr_info_t info;
        ASSERT_EQ(gdr_get_info(gdrHandle, gpuBufHandle.mh, &info), 0);
#ifdef DEBUG
        cout << "mappedDevicePtr: " << *map_d_ptr_p << endl;
        cout << "info.va: " << hex << info.va << dec << endl;
        cout << "info.mapped_size: " << info.mapped_size << endl;
        cout << "info.page_size: " << info.page_size << endl;
        cout << "info.mapped: " << info.mapped << endl;
        cout << "info.wc_mapping: " << info.wc_mapping << endl;
#endif
        // remember that mappings start on a 64KB boundary, so let's
        // calculate the offset from the head of the mapping to the
        // beginning of the buffer
        int off = info.va - gpuBufHandle.device_ptr;
        gpuBufHandle.host_ptr = (void*)((uintptr_t)gpuBufHandle.mappedDevicePtr + off);
#ifdef DEBUG
        cout << "page offset: " << off << endl;
        cout << "user-space pointer: " << *host_ptr << endl;

        cout << "CPU does gdr_copy_to_mapping and GPU writes back via cuMemHostAlloc'd buffer." << endl;
#endif
    } END_CHECK;
}

void cudaGPUMemFree(GPUBufferHandle& gpuBufHandle){
    gdr_t &gdrHandle = gpuDI.gdrHandle;
    
    BEGIN_CHECK {
        ASSERT_EQ(gdr_unmap(gdrHandle, gpuBufHandle.mh, gpuBufHandle.mappedDevicePtr, gpuBufHandle.allocationSize), 0);

        ASSERT_EQ(gdr_unpin_buffer(gdrHandle, gpuBufHandle.mh), 0);
    } END_CHECK;

    ASSERTDRV(gpu_mem_free(&gpuBufHandle.mhandle));
}

class GPUMultiBufferHandle : public GPUBufferHandle {
    GPUBufferHandle sharedGPUUserDataBuffers;
    size_t singleBufferSizeInBytes;
    public:

        __device__ __host__
        GPUBufferHandle& getSharedGPUUserDataBuffers(){
            return sharedGPUUserDataBuffers;
        }

        void setSingleBufferSizeInBytes(size_t buffer_size_in_bytes){
            this->singleBufferSizeInBytes = buffer_size_in_bytes;
        }

        __device__ __host__
        size_t getSingleBufferSizeInBytes(){
            return singleBufferSizeInBytes;
        }

        // __device__ __host__
        // void* getHostPtrSingleBuffer(int i){ // TODO guy fix this function if it's ever needed
        //     return sharedGPUUserDataBuffers.getHostPtr() + i * single_buffer_size_in_bytes;
        // }

        __device__ __host__
        CUdeviceptr getDevicePtrSingleBuffer(int i){
            return sharedGPUUserDataBuffers.getDevicePtr() + i * getSingleBufferSizeInBytes();
        }
};

void cudaGPUMultiBufferAlloc(GPUMultiBufferHandle& gpuMultiBufHandle, size_t num_buffers, size_t buffer_size_in_bytes)
{
    cudaGPUMemAlloc(gpuMultiBufHandle, num_buffers * sizeof(char*));
    gpuMultiBufHandle.setSingleBufferSizeInBytes(buffer_size_in_bytes);

    // Allocate a single large buffer for all data buffers
    cudaGPUMemAlloc(gpuMultiBufHandle.getSharedGPUUserDataBuffers(), num_buffers * buffer_size_in_bytes);

    // Get the host pointers for both buffers
    char** userBuffsHostPtr = (char**)gpuMultiBufHandle.getHostPtr();
    char* dataBuffersHostPtr = (char*)gpuMultiBufHandle.getSharedGPUUserDataBuffers().getHostPtr();
    // Assign the pointers within user buffer to the correct offsets in the data buffer
    for (int i = 0; i < num_buffers; i++)
    {
        userBuffsHostPtr[i] = dataBuffersHostPtr + i * buffer_size_in_bytes;
    }
}

void cudaGPUMultiBufferFree(GPUMultiBufferHandle& gpuMultiBufHandle){
    cudaGPUMemFree(gpuMultiBufHandle);
    cudaGPUMemFree(gpuMultiBufHandle.getSharedGPUUserDataBuffers());
}