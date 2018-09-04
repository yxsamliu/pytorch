#include "hip/hip_runtime.h"
#include "caffe2/operators/tanh_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void
TanhGradientHIPKernel(const int N, const T* dY, const T* Y, T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * (T(1) - __ldg(Y + i) * __ldg(Y + i));
#else
    dX[i] = dY[i] * (T(1) - Y[i] * Y[i]);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool TanhGradientFunctor<HIPContext>::Forward(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const T* Y,
    const T* dY,
    T* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
 hipLaunchKernelGGL( TanhGradientHIPKernel<T>
      , dim3(CAFFE_GET_BLOCKS(size)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), size, dY, Y, dX);
  return true;
}

REGISTER_HIP_OPERATOR(
    Tanh,
    UnaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        TanhFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    TanhGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        TanhGradientFunctor<HIPContext>>);

} // namespace caffe2
