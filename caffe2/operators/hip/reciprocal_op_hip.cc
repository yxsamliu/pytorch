#include "hip/hip_runtime.h"
#include "caffe2/operators/reciprocal_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void
ReciprocalGradientHIPKernel(const int N, const T* dY, const T* Y, T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * (-__ldg(Y + i) * __ldg(Y + i));
#else
    dX[i] = dY[i] * (-Y[i] * Y[i]);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool ReciprocalGradientFunctor<HIPContext>::Forward(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const T* Y,
    const T* dY,
    T* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
 hipLaunchKernelGGL( ReciprocalGradientHIPKernel<T>
      , dim3(CAFFE_GET_BLOCKS(size)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), size, dY, Y, dX);
  return true;
}

REGISTER_HIP_OPERATOR(
    Reciprocal,
    UnaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        ReciprocalFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    ReciprocalGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        ReciprocalGradientFunctor<HIPContext>>);

} // namespace caffe2
