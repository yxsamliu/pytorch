#include "hip/hip_runtime.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/channel_shuffle_op.h"

namespace caffe2 {

__global__ void ChannelShuffleNCHWKernel(
    const int N,
    const int S,
    const int C,
    const int G,
    const int K,
    const float* Xdata,
    float* Ydata) {
  HIP_1D_KERNEL_LOOP(i, N) {
    const int out_s = i % S;
    const int i_2 = i / S;
    const int out_c = i_2 % C;
    const int n = i_2 / C;

    const int g = out_c % G;
    const int k = out_c / G;
    const int in_c = k + K * g;
    Ydata[out_s + S * out_c + S * C * n] = Xdata[out_s + S * in_c + S * C * n];
  }
}

__global__ void ChannelShuffleNHWCKernel(
    const int N,
    const int G,
    const int K,
    const float* Xdata,
    float* Ydata) {
  HIP_1D_KERNEL_LOOP(i, N) {
    const int out_g = i % G;
    const int i_2 = i / G;
    const int out_k = i_2 % K;
    const int n = i_2 / K;

    const int in_c = out_k + K * out_g;
    Ydata[out_g + G * out_k + G * K * n] = Xdata[in_c + G * K * n];
  }
}

template <>
bool ChannelShuffleOp<float, HIPContext>::RunOnDeviceWithOrderNCHW() {
  const auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  const auto C = X.dim32(1);
  const auto G = this->group_;
  CAFFE_ENFORCE(C % G == 0, "");
  const auto K = C / G;
  const auto S = X.dim32(2) * X.dim32(3);
 hipLaunchKernelGGL( ChannelShuffleNCHWKernel, 
      dim3(CAFFE_GET_BLOCKS(X.size())),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(X.size()), static_cast<const int>(S), static_cast<const int>(C), static_cast<const int>(G), static_cast<const int>(K), X.data<float>(), Y->template mutable_data<float>());
  return true;
}

template <>
bool ChannelShuffleOp<float, HIPContext>::RunOnDeviceWithOrderNHWC() {
  const auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);
  const auto C = X.dim32(3);
  const auto G = this->group_;
  CAFFE_ENFORCE(C % G == 0, "");
  const auto K = C / G;
 hipLaunchKernelGGL( ChannelShuffleNHWCKernel, 
      dim3(CAFFE_GET_BLOCKS(X.size())),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(X.size()), static_cast<const int>(G), static_cast<const int>(K), X.data<float>(), Y->template mutable_data<float>());
  return true;
}

template <>
bool ChannelShuffleGradientOp<float, HIPContext>::RunOnDeviceWithOrderNCHW() {
  const auto& dY = Input(0);
  auto* dX = Output(0);
  dX->ResizeLike(dY);
  const auto C = dY.dim32(1);
  const auto G = this->group_;
  CAFFE_ENFORCE(C % G == 0, "");
  const auto K = C / G;
  const auto S = dY.dim32(2) * dY.dim32(3);
 hipLaunchKernelGGL( ChannelShuffleNCHWKernel, 
      dim3(CAFFE_GET_BLOCKS(dY.size())),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(dY.size()),
      static_cast<const int>(S),
      static_cast<const int>(C),
      static_cast<const int>(K),
      static_cast<const int>(G),
      dY.data<float>(),
      dX->template mutable_data<float>());
  return true;
}

template <>
bool ChannelShuffleGradientOp<float, HIPContext>::RunOnDeviceWithOrderNHWC() {
  const auto& dY = Input(0);
  auto* dX = Output(0);
  dX->ResizeLike(dY);
  const auto C = dY.dim32(3);
  const auto G = this->group_;
  CAFFE_ENFORCE(C % G == 0, "");
  const auto K = C / G;
 hipLaunchKernelGGL( ChannelShuffleNHWCKernel, 
      dim3(CAFFE_GET_BLOCKS(dY.size())),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(dY.size()), static_cast<const int>(K), static_cast<const int>(G), dY.data<float>(), dX->template mutable_data<float>());
  return true;
}

REGISTER_HIP_OPERATOR(ChannelShuffle, ChannelShuffleOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    ChannelShuffleGradient,
    ChannelShuffleGradientOp<float, HIPContext>);

} // namespace caffe2
