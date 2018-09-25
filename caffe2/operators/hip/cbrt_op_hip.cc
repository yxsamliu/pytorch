#include "hip/hip_runtime.h"
#include "caffe2/operators/cbrt_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void
CbrtGradientHIPKernel(const int N, const T* dY, const T* Y, T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    dX[i] = __ldg(dY + i) / (__ldg(Y + i) * __ldg(Y + i) * T(3));
#else
    dX[i] = dY[i] / (Y[i] * Y[i] * T(3));
#endif
  }
}

} // namespace

template <>
template <typename T>
bool CbrtGradientFunctor<HIPContext>::Forward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& /* Y_dims */,
    const T* dY,
    const T* Y,
    T* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
 hipLaunchKernelGGL( CbrtGradientHIPKernel<T>
      , dim3(CAFFE_GET_BLOCKS(size)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), size, dY, Y, dX);
  return true;
}

REGISTER_HIP_OPERATOR(
    Cbrt,
    UnaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        CbrtFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    CbrtGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        CbrtGradientFunctor<HIPContext>>);

} // namespace caffe2
