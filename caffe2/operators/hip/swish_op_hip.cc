#include "hip/hip_runtime.h"
#include "caffe2/operators/swish_op.h"

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void SwishHIPKernel(const int N, const T* X, T* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    Y[i] = __ldg(X + i) / (T(1) + exp(-__ldg(X + i)));
#else
    Y[i] = X[i] / (T(1) + exp(-X[i]));
#endif
  }
}

template <typename T>
__global__ void SwishGradientHIPKernel(
    const int N,
    const T* X,
    const T* Y,
    const T* dY,
    T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    dX[i] = __ldg(dY + i) *
        (__ldg(Y + i) + (T(1) - __ldg(Y + i)) / (T(1) + exp(-__ldg(X + i))));
#else
    dX[i] = dY[i] * (Y[i] + (T(1) - Y[i]) / (T(1) + exp(-X[i])));
#endif
  }
}

} // namespace

template <>
template <typename T>
bool SwishFunctor<HIPContext>::
operator()(const int N, const T* X, T* Y, HIPContext* context) const {
 hipLaunchKernelGGL( SwishHIPKernel<T>
      , dim3(CAFFE_GET_BLOCKS(N)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), N, X, Y);
  return true;
}

template <>
template <typename T>
bool SwishGradientOp<HIPContext>::DoRunWithType() {
  auto& Xin = Input(X);
  auto& Yin = Input(Y);
  auto& DYin = Input(DY);
  auto* DXout = Output(DX);
  CAFFE_ENFORCE_EQ(Xin.size(), Yin.size());
  CAFFE_ENFORCE_EQ(DYin.size(), Yin.size());
  DXout->ResizeLike(Yin);

  const int n = Xin.size();
  const T* x = Xin.template data<T>();
  const T* y = Yin.template data<T>();
  const T* dy = DYin.template data<T>();
  T* dx = DXout->template mutable_data<T>();
 hipLaunchKernelGGL( SwishGradientHIPKernel<T>
      , dim3(CAFFE_GET_BLOCKS(n)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context_.hip_stream(), n, x, y, dy, dx);
  return true;
}

template <>
bool SwishGradientOp<HIPContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, double>>::call(this, Input(X));
}

REGISTER_HIP_OPERATOR(
    Swish,
    UnaryElementwiseOp<
        TensorTypes<float, double>,
        HIPContext,
        SwishFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(SwishGradient, SwishGradientOp<HIPContext>);

} // namespace caffe2
