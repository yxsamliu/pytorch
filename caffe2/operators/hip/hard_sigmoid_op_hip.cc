#include "hip/hip_runtime.h"
#include "caffe2/operators/hard_sigmoid_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void HardSigmoidHIPKernel(
    const int N,
    const T alpha,
    const T beta,
    const T* X,
    T* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    Y[i] = max(T(0), min(T(1), alpha * __ldg(X + i) + beta));
#else
    Y[i] = max(T(0), min(T(1), alpha * X[i] + beta));
#endif
  }
}

template <typename T>
__global__ void HardSigmoidGradientHIPKernel(
    const int N,
    const T alpha,
    const T* dY,
    const T* Y,
    T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    dX[i] = (__ldg(Y + i) > T(0) && __ldg(Y + i) < T(1)) ? __ldg(dY + i) * alpha
                                                         : T(0);
#else
    dX[i] = (Y[i] > T(0) && Y[i] < T(1)) ? dY[i] * alpha : T(0);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool HardSigmoidFunctor<HIPContext>::
operator()(const int N, const T* X, T* Y, HIPContext* context) const {
 hipLaunchKernelGGL( HardSigmoidHIPKernel<T>
      , dim3(CAFFE_GET_BLOCKS(N)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), N, alpha, beta, X, Y);
  return true;
}

template <>
template <typename T>
bool HardSigmoidGradientFunctor<HIPContext>::Forward(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const T* Y,
    const T* dY,
    T* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
 hipLaunchKernelGGL( HardSigmoidGradientHIPKernel<T>
      , dim3(CAFFE_GET_BLOCKS(size)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), size, alpha, dY, Y, dX);
  return true;
}

REGISTER_HIP_OPERATOR(
    HardSigmoid,
    UnaryElementwiseWithArgsOp<
        TensorTypes<float>,
        HIPContext,
        HardSigmoidFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    HardSigmoidGradient,
    BinaryElementwiseWithArgsOp<
        TensorTypes<float>,
        HIPContext,
        HardSigmoidGradientFunctor<HIPContext>>);

} // namespace caffe2
