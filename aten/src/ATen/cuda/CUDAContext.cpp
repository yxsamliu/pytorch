#include "ATen/cuda/CUDAContext.h"
#include "THC/THCGeneral.hpp"

namespace at { namespace cuda {

/* Device info */
int64_t getNumGPUs() {
  int count;
  AT_CUDA_CHECK(hipGetDeviceCount(&count));
  return count;
}

int64_t current_device() {
  int cur_device;
  AT_CUDA_CHECK(hipGetDevice(&cur_device));
  return cur_device;
}

void set_device(int64_t device) {
  AT_CUDA_CHECK(hipSetDevice((int)device));
}

int warp_size() {
  return getCurrentDeviceProperties()->warpSize;
}

hipDeviceProp_t* getCurrentDeviceProperties() {
  return THCState_getCurrentDeviceProperties(at::globalContext().getTHCState());
}

hipDeviceProp_t* getDeviceProperties(int64_t device) {
  return THCState_getDeviceProperties(at::globalContext().getTHCState(), (int)device);
}

/* Streams */
CUDAStream getStreamFromPool(
  const bool isHighPriority
, int64_t device) {
  return CUDAStream(detail::CUDAStream_getStreamFromPool(isHighPriority, device));
}

CUDAStream getDefaultCUDAStream(int64_t device) {
  return CUDAStream(detail::CUDAStream_getDefaultStream(device));
}
CUDAStream getCurrentCUDAStream(int64_t device) {
  return CUDAStream(detail::CUDAStream_getCurrentStream(device));
}

void setCurrentCUDAStream(CUDAStream stream) {
  detail::CUDAStream_setStream(stream.internals());
}
void uncheckedSetCurrentCUDAStream(CUDAStream stream) {
  detail::CUDAStream_uncheckedSetStream(stream.internals());
}

Allocator* getCUDADeviceAllocator() {
  return at::globalContext().getTHCState()->hipDeviceAllocator;
}

/* Handles */
hipsparseHandle_t getCurrentCUDASparseHandle() {
  return THCState_getCurrentSparseHandle(at::globalContext().getTHCState());
}

rocblas_handle getCurrentCUDABlasHandle() {
  return THCState_getCurrentBlasHandle(at::globalContext().getTHCState());
}

} // namespace cuda

} // namespace at
