#include "hip/hip_runtime.h"
#include "caffe2/operators/arg_ops.h"

#include <limits>

#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

#include "caffe2/core/hip/common_hip.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/utils/fixed_divisor.h"

namespace caffe2 {

namespace {

template <typename K, typename V>
using KeyValuePair = cub::KeyValuePair<K, V>;

template <typename K, typename V>
using BlockReduce =
    cub::BlockReduce<KeyValuePair<K, V>, CAFFE_HIP_NUM_THREADS>;

template <typename T, class Reducer>
__global__ void ComputeArgHIPKernel(
    const int outer_size,
    const int inner_size,
    const FixedDivisor<int> stride,
    const Reducer reducer,
    const T init,
    const T* X,
    int64_t* Y) {
  __shared__ typename BlockReduce<int, T>::TempStorage temp_storage;
  const int d = stride.d();
  for (int idx = blockIdx.x; idx < outer_size; idx += gridDim.x) {
    int i;
    int j;
    stride.DivMod(idx, &i, &j);
    KeyValuePair<int, T> kv = {-1, init};
    for (int k = threadIdx.x; k < inner_size; k += blockDim.x) {
      kv = reducer({k, X[i * inner_size * d + k * d + j]}, kv);
    }
    kv = BlockReduce<int, T>(temp_storage).Reduce(kv, reducer);
    if (threadIdx.x == 0) {
      Y[idx] = static_cast<int64_t>(kv.key);
    }
    __syncthreads();
  }
}

} // namespace

template <>
template <typename T>
bool ArgMaxReducer<HIPContext>::operator()(
    const int prev_size,
    const int next_size,
    const int n,
    const T* X,
    int64_t* Y,
    HIPContext* context) const {
  const int outer_size = prev_size * next_size;
  const FixedDivisor<int> stride(next_size);
 hipLaunchKernelGGL( ComputeArgHIPKernel, 
      dim3(::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(), 
      outer_size,
      n,
      stride,
      cub::ArgMax(),
      std::numeric_limits<T>::lowest(),
      X,
      Y);
  return true;
}

template <>
template <typename T>
bool ArgMinReducer<HIPContext>::operator()(
    const int prev_size,
    const int next_size,
    const int n,
    const T* X,
    int64_t* Y,
    HIPContext* context) const {
  const int outer_size = prev_size * next_size;
  const FixedDivisor<int> stride(next_size);
 hipLaunchKernelGGL( ComputeArgHIPKernel, 
      dim3(::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(), 
      outer_size,
      n,
      stride,
      cub::ArgMin(),
      std::numeric_limits<T>::max(),
      X,
      Y);
  return true;
}

REGISTER_HIP_OPERATOR(ArgMax, ArgOp<HIPContext, ArgMaxReducer<HIPContext>>);
REGISTER_HIP_OPERATOR(ArgMin, ArgOp<HIPContext, ArgMinReducer<HIPContext>>);

} // namespace caffe2
