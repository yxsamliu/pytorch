#include "hip/hip_runtime.h"
#include "caffe2/operators/sigmoid_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void SigmoidHIPKernel(const int N, const T* X, T* Y);

template <>
__global__ void
SigmoidHIPKernel<float>(const int N, const float* X, float* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    Y[i] = 1.0f / (1.0f + expf(-__ldg(X + i)));
#else
    Y[i] = 1.0f / (1.0f + expf(-X[i]));
#endif
  }
}

template <typename T>
__global__ void
SigmoidGradientHIPKernel(const int N, const T* dY, const T* Y, T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * __ldg(Y + i) * (T(1) - __ldg(Y + i));
#else
    dX[i] = dY[i] * Y[i] * (T(1) - Y[i]);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool SigmoidFunctor<HIPContext>::
operator()(const int N, const T* X, T* Y, HIPContext* context) const {
 hipLaunchKernelGGL( SigmoidHIPKernel<T>
      , dim3(CAFFE_GET_BLOCKS(N)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), N, X, Y);
  return true;
}

template <>
template <typename T>
bool SigmoidGradientFunctor<HIPContext>::Forward(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const T* Y,
    const T* dY,
    T* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
 hipLaunchKernelGGL( SigmoidGradientHIPKernel<T>
      , dim3(CAFFE_GET_BLOCKS(size)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), size, dY, Y, dX);
  return true;
}

REGISTER_HIP_OPERATOR(
    Sigmoid,
    UnaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        SigmoidFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    SigmoidGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        SigmoidGradientFunctor<HIPContext>>);

} // namespace caffe2
