#include "hip/hip_runtime.h"
#include "caffe2/operators/tan_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

template <typename T>
inline __host__ __device__ T Square(const T& x) {
  return x * x;
}

template <typename T>
__global__ void
TanGradientHIPKernel(const int N, const T* dY, const T* X, T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    dX[i] = __ldg(dY + i) / Square(cos(__ldg(X + i)));
#else
    dX[i] = dY[i] / Square(cos(X[i]));
#endif
  }
}

template <>
template <typename T>
bool TanGradientFunctor<HIPContext>::Forward(
    const std::vector<int>& X_dims,
    const std::vector<int>& /* dY_dims */,
    const T* X,
    const T* dY,
    T* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
 hipLaunchKernelGGL( TanGradientHIPKernel<T>
      , dim3(CAFFE_GET_BLOCKS(size)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), size, dY, X, dX);
  return true;
}

REGISTER_HIP_OPERATOR(
    Tan,
    UnaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        TanFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    TanGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        TanGradientFunctor<HIPContext>>);

} // namespace caffe2
