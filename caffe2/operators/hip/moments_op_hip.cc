#include "hip/hip_runtime.h"
#include "caffe2/operators/moments_op.h"

#include <array>
#include <functional>

#include "caffe2/core/hip/context_hip.h"
#include "caffe2/utils/fixed_divisor.h"

namespace caffe2 {

namespace {

template <typename T, int D>
__global__ void ComputeMomentsGradientHIPKernel(
    const int X_size,
    const SimpleArray<int, D> Y_strides,
    const SimpleArray<FixedDivisor<int>, D> X_dims,
    const T scale,
    const T* dmean,
    const T* dvariance,
    const T* X,
    const T* mean,
    T* dX) {
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
    dX[X_index] =
        (__ldg(dmean + Y_index) +
         static_cast<T>(2) * (__ldg(X + X_index) - __ldg(mean + Y_index)) *
             __ldg(dvariance + Y_index)) *
        scale;
#else
    dX[X_index] = (dmean[Y_index] +
                   static_cast<T>(2) * (X[X_index] - mean[Y_index]) *
                       dvariance[Y_index]) *
        scale;
#endif
  }
}

template <typename T, int D>
void ComputeMomentsGradientHIPImpl(
    const int* Y_dims,
    const int* X_dims,
    const T* dmean,
    const T* dvariance,
    const T* X,
    const T* mean,
    T* dX,
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
  const int Y_size =
      std::accumulate(Y_dims, Y_dims + D, 1, std::multiplies<int>());
  const int X_size =
      std::accumulate(X_dims, X_dims + D, 1, std::multiplies<int>());
  const T scale = static_cast<T>(Y_size) / static_cast<T>(X_size);
 hipLaunchKernelGGL( ComputeMomentsGradientHIPKernel<T, D>
      , dim3(CAFFE_GET_BLOCKS(X_size)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), 
          X_size,
          Y_strides_array,
          X_dims_array,
          scale,
          dmean,
          dvariance,
          X,
          mean,
          dX);
}

} // namespace

template <>
bool MomentsGradientOp<float, HIPContext>::Compute(
    const std::vector<int>& dY_dims,
    const std::vector<int>& dX_dims,
    const float* dmean_data,
    const float* dvariance_data,
    const float* X_data,
    const float* mean_data,
    float* dX_data) {
  const int ndim = dY_dims.size();
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
      ndim,
      ComputeMomentsGradientHIPImpl,
      float,
      dY_dims.data(),
      dX_dims.data(),
      dmean_data,
      dvariance_data,
      X_data,
      mean_data,
      dX_data,
      &context_);
  return true;
}

REGISTER_HIP_OPERATOR(Moments, MomentsOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(MomentsGradient, MomentsGradientOp<float, HIPContext>);

} // namespace caffe2
