#include "hip/hip_runtime.h"
#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

#include "ATen/cuda/CUDAContext.h"
#include "hip/hip_runtime.h"

namespace at {
namespace native {

Scalar _local_scalar_dense_cuda(const Tensor& self) {
  Scalar r;
  AT_DISPATCH_ALL_TYPES_AND_HALF(
      self.type(), "_local_scalar_dense_cuda", [&] {
        scalar_t value;
        hipStream_t stream = at::cuda::getCurrentCUDAStream();
        AT_CUDA_CHECK(hipMemcpyAsync(&value, self.data<scalar_t>(), sizeof(scalar_t), hipMemcpyDeviceToHost, stream));
        AT_CUDA_CHECK(hipStreamSynchronize(stream));
        r = Scalar(value);
      });
  return r;
}

}} // at::native
