#pragma once

typedef struct CUDAStreamInternals THCStream;

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

// RAII wrapper for CUDA streams.
//
// This wrapper uses THCStream instead of hipStream_t because we need
// to interact with the THC API for selecting the current stream.
// Doing this without having a THCStream pointer is cumbersome.
//
class CUDAStream {
 public:
  CUDAStream(THCStream* stream) : stream_(stream) {}

  CUDAStream() : CUDAStream(nullptr) {}

  ~CUDAStream();

  static CUDAStream create();

  // Must not be copyable.
  CUDAStream& operator=(const CUDAStream&) = delete;
  CUDAStream(const CUDAStream&) = delete;

  // Must be move constructable.
  CUDAStream(CUDAStream&& other) {
    std::swap(stream_, other.stream_);
  }

  // Must be move assignable.
  CUDAStream& operator=(CUDAStream&& other) {
    std::swap(stream_, other.stream_);
    return *this;
  }

  hipStream_t getStream() const;

  THCStream* getTHCStream() {
    return stream_;
  }

 protected:
  THCStream* stream_;
};

} // namespace c10d
