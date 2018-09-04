#include "hip/hip_runtime.h"
#include "caffe2/operators/relu_n_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void
ReluNHIPKernel(const int N, const T threshold, const T* X, T* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    Y[i] = __ldg(X + i) > 0
        ? (__ldg(X + i) < threshold ? __ldg(X + i) : threshold)
        : T(0);
#else
    Y[i] = X[i] > 0 ? (X[i] < threshold ? X[i] : threshold) : T(0);
#endif
  }
}

template <typename T>
__global__ void ReluNGradientHIPKernel(
    const int N,
    const T threshold,
    const T* dY,
    const T* Y,
    T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    dX[i] = (__ldg(Y + i) > 0 && __ldg(Y + i) < threshold) ? dY[i] : T(0);
#else
    dX[i] = (Y[i] > 0 && Y[i] < threshold) ? dY[i] : T(0);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool ReluNFunctor<HIPContext>::
operator()(const int N, const T* X, T* Y, HIPContext* context) const {
 hipLaunchKernelGGL( ReluNHIPKernel<T>
      , dim3(CAFFE_GET_BLOCKS(N)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), N, n, X, Y);
  return true;
}

template <>
template <typename T>
bool ReluNGradientFunctor<HIPContext>::Forward(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const T* Y,
    const T* dY,
    T* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
 hipLaunchKernelGGL( ReluNGradientHIPKernel<T>
      , dim3(CAFFE_GET_BLOCKS(size)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), size, n, dY, Y, dX);
  return true;
}

REGISTER_HIP_OPERATOR(
    ReluN,
    UnaryElementwiseWithArgsOp<
        TensorTypes<float>,
        HIPContext,
        ReluNFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    ReluNGradient,
    BinaryElementwiseWithArgsOp<
        TensorTypes<float>,
        HIPContext,
        ReluNGradientFunctor<HIPContext>>);

} // namespace caffe2
