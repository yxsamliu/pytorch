#include "hip/hip_runtime.h"
#include "caffe2/operators/sin_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void
SinGradientHIPKernel(const int N, const T* dY, const T* X, T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * cos(__ldg(X + i));
#else
    dX[i] = dY[i] * cos(X[i]);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool SinGradientFunctor<HIPContext>::Forward(
    const std::vector<int>& X_dims,
    const std::vector<int>& /* dY_dims */,
    const T* X,
    const T* dY,
    T* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
 hipLaunchKernelGGL( SinGradientHIPKernel<T>
      , dim3(CAFFE_GET_BLOCKS(size)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), size, dY, X, dX);
  return true;
}

REGISTER_HIP_OPERATOR(
    Sin,
    UnaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        SinFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    SinGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        SinGradientFunctor<HIPContext>>);

} // namespace caffe2
