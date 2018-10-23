#pragma once

#include <algorithm>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

namespace c10d {

// RAII wrapper for CUDA events.
class CUDAEvent {
 public:
  CUDAEvent(hipEvent_t event, int device) : device_(device), event_(event) {}

  CUDAEvent() : CUDAEvent(nullptr, 0) {}

  ~CUDAEvent() noexcept(false);

  static CUDAEvent create(unsigned int flags = hipEventDefault);

  // Must not be copyable.
  CUDAEvent& operator=(const CUDAEvent&) = delete;
  CUDAEvent(const CUDAEvent&) = delete;

  // Must be move constructable.
  CUDAEvent(CUDAEvent&& other) {
    std::swap(event_, other.event_);
    std::swap(device_, other.device_);
  }

  // Must be move assignable.
  CUDAEvent& operator=(CUDAEvent&& other) {
    std::swap(event_, other.event_);
    std::swap(device_, other.device_);
    return *this;
  }

  hipEvent_t getEvent() const {
    return event_;
  }

  int getDevice() const {
    return device_;
  }

 protected:
  int device_;
  hipEvent_t event_;
};

} // namespace c10d
