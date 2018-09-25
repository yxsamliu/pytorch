#include "hip/hip_runtime.h"
#include "caffe2/operators/cosh_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

__global__ void CoshGradientHIPKernel(
    const int N,
    const float* dY,
    const float* X,
    float* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * sinhf(__ldg(X + i));
#else
    dX[i] = dY[i] * sinhf(X[i]);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool CoshGradientFunctor<HIPContext>::Forward(
    const std::vector<int>& /* dY_dims */,
    const std::vector<int>& X_dims,
    const T* dY,
    const T* X,
    T* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
 hipLaunchKernelGGL( CoshGradientHIPKernel, 
      dim3(CAFFE_GET_BLOCKS(size)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(), size, dY, X, dX);
  return true;
}

REGISTER_HIP_OPERATOR(
    Cosh,
    UnaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        CoshFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    CoshGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        CoshGradientFunctor<HIPContext>>);

} // namespace caffe2
