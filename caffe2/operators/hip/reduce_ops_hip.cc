#include "hip/hip_runtime.h"
#include "caffe2/operators/reduce_ops.h"

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe2/core/hip/common_hip.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/utils/fixed_divisor.h"

namespace caffe2 {

namespace {

template <typename T, int D>
__global__ void ComputeReduceMinMaxGradientHIPKernel(
    const int X_size,
    const SimpleArray<int, D> Y_strides,
    const SimpleArray<FixedDivisor<int>, D> X_dims,
    const T* dY_data,
    const T* X_data,
    const T* Y_data,
    T* dX_data) {
  HIP_1D_KERNEL_LOOP(X_index, X_size) {
    int Y_index = 0;
    int X_index_val = X_index;
#pragma unroll
    for (int i = D - 1; i >= 0; --i) {
      int d;
      X_dims.data[i].DivMod(X_index_val, &X_index_val, &d);
      Y_index += d * Y_strides.data[i];
    }
#if __HIP_ARCH__ >= 350
    dX_data[X_index] = __ldg(Y_data + Y_index) == __ldg(X_data + X_index)
        ? __ldg(dY_data + Y_index)
        : T(0);
#else
    dX_data[X_index] =
        Y_data[Y_index] == X_data[X_index] ? dY_data[Y_index] : T(0);
#endif
  }
}

template <typename T, int D>
void ComputeReduceMinMaxGradientHIPImpl(
    const int* Y_dims,
    const int* X_dims,
    const T* dY_data,
    const T* X_data,
    const T* Y_data,
    T* dX_data,
    HIPContext* context) {
  SimpleArray<int, D> Y_strides_array;
  SimpleArray<FixedDivisor<int>, D> X_dims_array;
  int cur_stride = 1;
  for (int i = D - 1; i >= 0; --i) {
    if (X_dims[i] == 0) {
      return;
    }
    Y_strides_array.data[i] = Y_dims[i] == 1 ? 0 : cur_stride;
    X_dims_array.data[i] = FixedDivisor<int>(X_dims[i]);
    cur_stride *= Y_dims[i];
  }
  const int X_size =
      std::accumulate(X_dims, X_dims + D, 1, std::multiplies<int>());
 hipLaunchKernelGGL( ComputeReduceMinMaxGradientHIPKernel<T, D>
      , dim3(CAFFE_GET_BLOCKS(X_size)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), 
          X_size,
          Y_strides_array,
          X_dims_array,
          dY_data,
          X_data,
          Y_data,
          dX_data);
}

} // namespace

template <>
template <typename T>
bool MinReducer<HIPContext>::Backward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& dX_dims,
    const T* dY_data,
    const T* X_data,
    const T* Y_data,
    T* dX_data,
    HIPContext* context) const {
  const int ndim = dY_dims.size();
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
      ndim,
      ComputeReduceMinMaxGradientHIPImpl,
      T,
      dY_dims.data(),
      dX_dims.data(),
      dY_data,
      X_data,
      Y_data,
      dX_data,
      context);
  return true;
}

template <>
template <typename T>
bool MaxReducer<HIPContext>::Backward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& dX_dims,
    const T* dY_data,
    const T* X_data,
    const T* Y_data,
    T* dX_data,
    HIPContext* context) const {
  const int ndim = dY_dims.size();
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
      ndim,
      ComputeReduceMinMaxGradientHIPImpl,
      T,
      dY_dims.data(),
      dX_dims.data(),
      dY_data,
      X_data,
      Y_data,
      dX_data,
      context);
  return true;
}

REGISTER_HIP_OPERATOR(
    ReduceMin,
    ReduceOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        HIPContext,
        MinReducer<HIPContext>>);
REGISTER_HIP_OPERATOR(
    ReduceMinGradient,
    ReduceGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        HIPContext,
        MinReducer<HIPContext>>);

REGISTER_HIP_OPERATOR(
    ReduceMax,
    ReduceOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        HIPContext,
        MaxReducer<HIPContext>>);
REGISTER_HIP_OPERATOR(
    ReduceMaxGradient,
    ReduceGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        HIPContext,
        MaxReducer<HIPContext>>);

REGISTER_HIP_OPERATOR(
    ReduceSum,
    ReduceOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        HIPContext,
        SumReducer<HIPContext>>);
REGISTER_HIP_OPERATOR(
    ReduceSumGradient,
    ReduceGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        HIPContext,
        SumReducer<HIPContext>>);

REGISTER_HIP_OPERATOR(
    ReduceMean,
    ReduceOp<TensorTypes<float>, HIPContext, MeanReducer<HIPContext>>);
REGISTER_HIP_OPERATOR(
    ReduceMeanGradient,
    ReduceGradientOp<
        TensorTypes<float>,
        HIPContext,
        MeanReducer<HIPContext>>);

} // namespace caffe2
