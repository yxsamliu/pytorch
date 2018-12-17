#include "THCAllocator.h"

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
