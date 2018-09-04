#include "hip/hip_runtime.h"
#include "caffe2/operators/logit_op.h"

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void LogitKernel(const int N, const T* X, const float eps, T* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = fminf(X[i], (T(1) - eps));
    Y[i] = fmaxf(Y[i], eps);
    Y[i] = logf(Y[i] / (T(1) - Y[i]));
  }
}

template <typename T>
__global__ void LogitGradientKernel(
    const int N,
    const T* X,
    const T* dY,
    const float eps,
    T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
    dX[i] = (X[i] < eps || X[i] > T(1) - eps) ? T(0)
                                              : (dY[i] / X[i] / (T(1) - X[i]));
  }
}

} // namespace

template <>
template <typename T>
bool LogitFunctor<HIPContext>::
operator()(const int N, const T* X, T* Y, HIPContext* context) const {
 hipLaunchKernelGGL( LogitKernel<T>
      , dim3(CAFFE_GET_BLOCKS(N)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), static_cast<const int>(N), X, eps_, Y);
  return true;
}

template <>
bool LogitGradientOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(X);
  int n = X.size();
 hipLaunchKernelGGL( LogitGradientKernel, 
      dim3(CAFFE_GET_BLOCKS(n)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(n),
      X.data<float>(),
      dY.data<float>(),
      eps_,
      dX->template mutable_data<float>());
  return true;
}

REGISTER_HIP_OPERATOR(
    Logit,
    UnaryElementwiseWithArgsOp<
        TensorTypes<float>,
        HIPContext,
        LogitFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(LogitGradient, LogitGradientOp<float, HIPContext>);

} // namespace caffe2
