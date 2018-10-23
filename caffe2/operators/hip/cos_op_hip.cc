#include "hip/hip_runtime.h"
#include "caffe2/operators/cos_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void
CosGradientHIPKernel(const int N, const T* dY, const T* X, T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    dX[i] = -__ldg(dY + i) * sin(__ldg(X + i));
#else
    dX[i] = -dY[i] * sin(X[i]);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool CosGradientFunctor<HIPContext>::Forward(
    const std::vector<int>& X_dims,
    const std::vector<int>& /* dY_dims */,
    const T* X,
    const T* dY,
    T* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
 hipLaunchKernelGGL( CosGradientHIPKernel, 
      dim3(CAFFE_GET_BLOCKS(size)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(), size, dY, X, dX);
  return true;
}

REGISTER_HIP_OPERATOR(
    Cos,
    UnaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        CosFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    CosGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        CosGradientFunctor<HIPContext>>);

} // namespace caffe2
