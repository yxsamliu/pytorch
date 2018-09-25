#include "hip/hip_runtime.h"
#include "caffe2/operators/order_switch_ops.h"
#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

__global__ void NHWC2NCHWKernel(
    const int N,
    const int HW,
    const int C,
    const float* X,
    float* Y) {
  HIP_1D_KERNEL_LOOP(i, N * HW * C) {
    const int c = i % C;
    const int hw = i / C % HW;
    const int n = i / C / HW;
    Y[(n * C + c) * HW + hw] = X[i];
  }
}

__global__ void NCHW2NHWCKernel(
    const int N,
    const int C,
    const int HW,
    const float* X,
    float* Y) {
  HIP_1D_KERNEL_LOOP(i, N * C * HW) {
    const int hw = i % HW;
    const int c = i / HW % C;
    const int n = i / C / HW;
    Y[(n * HW + hw) * C + c] = X[i];
  }
}

template <>
bool NHWC2NCHWOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);

  auto ndim = X.ndim();
  DCHECK_GE(ndim, 3);
  const int N = X.dim32(0), C = X.dim32(ndim - 1);
  vector<TIndex> Y_dims(ndim);
  Y_dims[0] = N;
  Y_dims[1] = C;
  size_t image_size = 1;
  for (auto i = 2; i < ndim; ++i) {
    Y_dims[i] = X.dim32(i - 1);
    image_size *= Y_dims[i];
  }
  Y->Resize(Y_dims);

 hipLaunchKernelGGL( NHWC2NCHWKernel, 
      dim3(CAFFE_GET_BLOCKS(X.size())),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(N), static_cast<const int>(image_size), static_cast<const int>(C), X.data<float>(), Y->template mutable_data<float>());
  return true;
}

template <>
bool NCHW2NHWCOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);

  auto ndim = X.ndim();
  DCHECK_GE(X.ndim(), 3);
  const int N = X.dim32(0), C = X.dim32(1);
  vector<TIndex> Y_dims(ndim);
  Y_dims[0] = N;
  size_t image_size = 1;
  for (auto i = 1; i < ndim - 1; ++i) {
    Y_dims[i] = X.dim32(i + 1);
    image_size *= Y_dims[i];
  }
  Y_dims[ndim - 1] = C;
  Y->Resize(Y_dims);

 hipLaunchKernelGGL( NCHW2NHWCKernel, 
      dim3(CAFFE_GET_BLOCKS(X.size())),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(N), static_cast<const int>(C), static_cast<const int>(image_size), X.data<float>(), Y->template mutable_data<float>());
  return true;
}


REGISTER_HIP_OPERATOR(NHWC2NCHW, NHWC2NCHWOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(NCHW2NHWC, NCHW2NHWCOp<float, HIPContext>);
}  // namespace caffe2
