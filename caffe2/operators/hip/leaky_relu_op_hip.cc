#include "hip/hip_runtime.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/leaky_relu_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {
namespace {
template <typename T>
__global__ void LeakyReluKernel(const int N, const T alpha, const T* X, T* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = X[i] >= 0 ? X[i] : X[i] * alpha;
  }
}

template <typename T>
__global__ void LeakyReluGradientKernel(
    const int N,
    const T alpha,
    const T* Y,
    const T* dY,
    T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
    dX[i] = Y[i] >= 0 ? dY[i] : dY[i] * alpha;
  }
}
} // namespace

template <>
bool LeakyReluOp<float, HIPContext>::RunOnDevice() {
  const auto& X = Input(0);
  CAFFE_ENFORCE_GT(X.size(), 0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
 hipLaunchKernelGGL( LeakyReluKernel, 
      dim3(CAFFE_GET_BLOCKS(X.size())),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(X.size()), alpha_, X.data<float>(), Y->template mutable_data<float>());
  return true;
}

template <>
bool LeakyReluGradientOp<float, HIPContext>::RunOnDevice() {
  const auto& Y = Input(0);
  const auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(Y);
  CAFFE_ENFORCE_EQ(Y.size(), dY.size());
 hipLaunchKernelGGL( LeakyReluGradientKernel, 
      dim3(CAFFE_GET_BLOCKS(Y.size())),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(Y.size()),
      alpha_,
      Y.data<float>(),
      dY.data<float>(),
      dX->template mutable_data<float>());
  return true;
}

REGISTER_HIP_OPERATOR(LeakyRelu, LeakyReluOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    LeakyReluGradient,
    LeakyReluGradientOp<float, HIPContext>);
} // namespace caffe2
