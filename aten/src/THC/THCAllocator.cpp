#include "THCAllocator.h"

static void THCudaHostDeleter(void* ptr) {
  THCudaCheck(hipHostFree(ptr));
}

struct THCudaHostAllocator : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    void* ptr = nullptr;
    if (size != 0) {
      THCudaCheck(hipHostMalloc(&ptr, size));
    }
    return {ptr, ptr, &THCudaHostDeleter, at::DeviceType::CPU};
  }
  at::DeleterFnPtr raw_deleter() const override {
    return &THCudaHostDeleter;
  }
};

static THCudaHostAllocator th_cuda_host_allocator;
at::Allocator* getTHCudaHostAllocator() {
  return &th_cuda_host_allocator;
}

THCIpcDeleter::~THCIpcDeleter() {
  int prev_device;
  THCudaCheck(hipGetDevice(&prev_device));
  THCudaCheck(hipSetDevice(device_));
  THCudaCheck(hipIpcCloseMemHandle(data_));
  THCudaCheck(hipSetDevice(prev_device));
}

void deleteTHCIpcDeleter(void* ptr) {
  delete static_cast<THCIpcDeleter*>(ptr);
}

at::DataPtr THCIpcDeleter::makeDataPtr(void* data, int device) {
  // The dynamic allocation here is a bit unfortunate
  int cur_device;
  THCudaCheck(hipGetDevice(&cur_device));
  auto* context = new THCIpcDeleter(data, device);
  return {data, context, &deleteTHCIpcDeleter, at::Device(at::DeviceType::CUDA, cur_device)};
}
