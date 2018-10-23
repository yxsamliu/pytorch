#include "hip/hip_runtime.h"
#include <cub/block/block_reduce.cuh>

#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/one_hot_ops.h"

namespace caffe2 {

__global__ void OneHotOpKernel(
    const int64_t batch_size,
    const int64_t index_size,
    const int64_t* indices,
    float* output) {
  HIP_1D_KERNEL_LOOP(i, batch_size) {
    output[i * index_size + indices[i]] = 1.;
  }
}

template <>
void OneHotOp<HIPContext>::DoOneHotOp(
    int64_t batch_size,
    int64_t index_size,
    const Tensor& indices,
    Tensor* output) {
  float* output_ptr = output->template mutable_data<float>();
  math::Set<float, HIPContext>(output->size(), 0., output_ptr, &context_);
 hipLaunchKernelGGL( OneHotOpKernel, 
      dim3(CAFFE_GET_BLOCKS(batch_size)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      batch_size, index_size, static_cast<const int64_t*>(indices.data<int64_t>()), output_ptr);
}

REGISTER_HIP_OPERATOR(OneHot, OneHotOp<HIPContext>);
} // namespace
