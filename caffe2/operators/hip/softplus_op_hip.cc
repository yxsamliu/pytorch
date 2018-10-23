#include "hip/hip_runtime.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/softplus_op.h"

namespace caffe2 {
namespace {
template <typename T>
__global__ void SoftplusKernel(const int N, const T* X, T* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = log(exp(X[i]) + 1.0f);
  }
}

template <typename T>
__global__ void
SoftplusGradientKernel(const int N, const T* Y, const T* dY, T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
    const float nexpY = exp(-Y[i]);
    dX[i] = dY[i] * (1 - nexpY);
  }
}
} // namespace

template <>
bool SoftplusOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_GT(X.size(), 0);
  Y->ResizeLike(X);
 hipLaunchKernelGGL( SoftplusKernel<float>
      , dim3(CAFFE_GET_BLOCKS(X.size())),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context_.hip_stream(), 
          static_cast<const int>(X.size()), X.data<float>(), Y->template mutable_data<float>());
  return true;
}

template <>
bool SoftplusGradientOp<float, HIPContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  DCHECK_GT(Y.size(), 0);
  DCHECK_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);
 hipLaunchKernelGGL( SoftplusGradientKernel<float>
      , dim3(CAFFE_GET_BLOCKS(Y.size())),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context_.hip_stream(), 
          static_cast<const int>(Y.size()),
          Y.data<float>(),
          dY.data<float>(),
          dX->template mutable_data<float>());
  return true;
}

REGISTER_HIP_OPERATOR(Softplus, SoftplusOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    SoftplusGradient,
    SoftplusGradientOp<float, HIPContext>);
} // namespace caffe2
