#pragma once

#include "ATen/cuda/Exceptions.h"

#include "hip/hip_runtime.h"

namespace at {
namespace cuda {

inline Device getDeviceFromPtr(void* ptr) {
  struct hipPointerAttribute_t attr;
  AT_CUDA_CHECK(hipPointerGetAttributes(&attr, ptr));
  return {DeviceType::CUDA, attr.device};
}

}} // namespace at::cuda
