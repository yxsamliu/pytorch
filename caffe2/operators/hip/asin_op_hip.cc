#include "hip/hip_runtime.h"
#include "caffe2/operators/asin_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

__global__ void AsinGradientHIPKernel(
    const int N,
    const float* dY,
    const float* X,
    float* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * rsqrtf(1.0f - __ldg(X + i) * __ldg(X + i));
#else
    dX[i] = dY[i] * rsqrtf(1.0f - X[i] * X[i]);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool AsinGradientFunctor<HIPContext>::Forward(
    const std::vector<int>& X_dims,
    const std::vector<int>& /* dY_dims */,
    const T* X,
    const T* dY,
    T* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
 hipLaunchKernelGGL( AsinGradientHIPKernel, 
      dim3(CAFFE_GET_BLOCKS(size)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(), size, dY, X, dX);
  return true;
}

REGISTER_HIP_OPERATOR(
    Asin,
    UnaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        AsinFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    AsinGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        AsinGradientFunctor<HIPContext>>);

} // namespace caffe2
