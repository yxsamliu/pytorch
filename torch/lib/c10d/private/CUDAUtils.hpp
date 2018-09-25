#pragma once

#include <sstream>
#include <stdexcept>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

#include <ATen/ATen.h>
#include <THC/THCStream.h>

#include <c10d/CUDAUtils.hpp>

#define C10D_CUDA_CHECK(condition)        \
  do {                                    \
    hipError_t error = (condition);      \
    if (error != hipSuccess) {           \
      std::stringstream ss;               \
      ss << "Error at: ";                 \
      ss << __FILE__;                     \
      ss << ":";                          \
      ss << __LINE__;                     \
      ss << ": ";                         \
      ss << hipGetErrorString(error);    \
      throw std::runtime_error(ss.str()); \
    }                                     \
  } while (0)

namespace c10d {

// THCStreamGuard is a RAII guard for selecting a THCStream.
//
// It sets both the current device to the stream's device and the
// current stream in the THC state.
//
class THCStreamGuard {
 public:
  explicit THCStreamGuard(THCState* state, CUDAStream& stream)
      : device_(THCStream_device(stream.getTHCStream())), state_(state) {
    at::DeviceGuard deviceGuard(device_);
    original_ = THCState_getStream(state_);
    THCStream_retain(original_);
    THCState_setStream(state_, stream.getTHCStream());
  }

  THCStreamGuard(THCStreamGuard&& other)
      : device_(other.device_), state_(nullptr), original_(nullptr) {
    std::swap(state_, other.state_);
    std::swap(original_, other.original_);
  }

  ~THCStreamGuard() {
    if (original_ != nullptr) {
      at::DeviceGuard deviceGuard(device_);
      THCState_setStream(state_, original_);
      THCStream_free(original_);
    }
  }

 private:
  const int device_;
  THCState* state_;
  THCStream* original_;
};

} // namespace c10d
