#include "hip/hip_runtime.h"
#include "caffe2/operators/cube_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void
CubeGradientHIPKernel(const int N, const T* dY, const T* X, T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    dX[i] = __ldg(dY + i) * __ldg(X + i) * __ldg(X + i) * T(3);
#else
    dX[i] = dY[i] * X[i] * X[i] * T(3);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool CubeGradientFunctor<HIPContext>::Forward(
    const std::vector<int>& dY_dims,
    const std::vector<int>& /* X_dims */,
    const T* dY,
    const T* X,
    T* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
 hipLaunchKernelGGL( CubeGradientHIPKernel<T>
      , dim3(CAFFE_GET_BLOCKS(size)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), size, dY, X, dX);
  return true;
}

REGISTER_HIP_OPERATOR(
    Cube,
    UnaryElementwiseOp<NumericTypes, HIPContext, CubeFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    CubeGradient,
    BinaryElementwiseOp<
        NumericTypes,
        HIPContext,
        CubeGradientFunctor<HIPContext>>);

} // namespace caffe2
