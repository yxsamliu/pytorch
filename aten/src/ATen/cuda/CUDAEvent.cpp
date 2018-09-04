#include "ATen/cuda/CUDAEvent.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/CUDAStream.h"
#include "ATen/cuda/Exceptions.h"
#include "ATen/core/Error.h"

#include <mutex>
#include <atomic>

// Internal implementation is entirely hidden
struct CUDAEventInternals {
  std::atomic<int> refcount;
  int64_t device; // Note: hipGetDevice works with int32_t, not int64_t
  hipEvent_t event;
};

namespace at {
namespace cuda {

namespace detail {

/*
* Pointer-based event API
*/
CUDAEventInternals* CUDAEvent_create(unsigned int flags) {
  std::unique_ptr<CUDAEventInternals> internals { new CUDAEventInternals() };
  internals->refcount = 1;
  internals->device = current_device();
  AT_CUDA_CHECK(hipEventCreateWithFlags(&internals->event, flags));
  return internals.release();
}

void CUDAEvent_retain(CUDAEventInternals* internals) {
  internals->refcount++;
}

void CUDAEvent_uncheckedFree(CUDAEventInternals* internals) {
  if (--internals->refcount) {
    hipEventDestroy(internals->event);
  }
}
hipEvent_t CUDAEvent_event(CUDAEventInternals* internals) {
  return internals->event;
}

int64_t CUDAEvent_device(CUDAEventInternals* internals) {
  return internals->device;
}

void CUDAEvent_record(CUDAEventInternals* internals, const CUDAStream& stream) {
  AT_CUDA_CHECK(hipEventRecord(internals->event, stream));
}

} // namespace detail

void CUDAEvent::record() const {
  record(getCurrentCUDAStream());
}

void CUDAEvent::record(const CUDAStream& stream) const {
  detail::CUDAEvent_record(internals_, stream);
}


} // namespace cuda
} // namespace at
