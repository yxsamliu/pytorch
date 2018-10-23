#include "hip/hip_runtime.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/dropout_op.h"

namespace caffe2 {

namespace {
__global__ void DropoutKernel(
    const int N,
    const float ratio,
    const float* Xdata,
    float* Ydata,
    bool* maskdata) {
  const float scale = 1. / (1. - ratio);
  HIP_1D_KERNEL_LOOP(i, N) {
    maskdata[i] = (Ydata[i] > ratio);
    Ydata[i] = Xdata[i] * scale * maskdata[i];
  }
}
} // namespace

template <>
bool DropoutOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->Resize(X.dims());
  if (is_test_) {
    if (Y != &X) {
      context_.CopySameDevice<float>(
          X.size(), X.data<float>(), Y->template mutable_data<float>());
    }
    return true;
  } else {
    // We do a simple trick here: since curand cannot generate random
    // boolean numbers, we will generate into dY and write the result to
    // mask.
    float* Ydata = Y->template mutable_data<float>();
    auto* mask = Output(1);
    mask->Resize(X.dims());
    CAFFE_ENFORCE(X.data<float>() != Ydata, "In-place GPU dropout is broken");
    HIPRAND_ENFORCE(
        hiprandGenerateUniform(context_.hiprand_generator(), Ydata, X.size()));
   hipLaunchKernelGGL( DropoutKernel, 
        dim3(CAFFE_GET_BLOCKS(X.size())),
        dim3(CAFFE_HIP_NUM_THREADS),
        0,
        context_.hip_stream(), 
        static_cast<const int>(X.size()),
        ratio_,
        X.data<float>(),
        Ydata,
        mask->template mutable_data<bool>());
    return true;
  }
}

namespace {
__global__ void DropoutGradientKernel(
    const int N,
    const float* dYdata,
    const bool* maskdata,
    const float scale,
    float* dXdata) {
  HIP_1D_KERNEL_LOOP(i, N) {
    dXdata[i] = dYdata[i] * maskdata[i] * scale;
  }
}
} // namespace

template <>
bool DropoutGradientOp<float, HIPContext>::RunOnDevice() {
  auto& dY = Input(0);
  auto* dX = Output(0);
  dX->Resize(dY.dims());
  if (is_test_) {
    if (dX != &dY) {
      context_.CopySameDevice<float>(
          dY.size(), dY.data<float>(), dX->template mutable_data<float>());
    }
    return true;
  } else {
    auto& mask = Input(1);
    CAFFE_ENFORCE_EQ(dY.size(), mask.size());
    const float scale = 1. / (1. - ratio_);
   hipLaunchKernelGGL( DropoutGradientKernel, 
        dim3(CAFFE_GET_BLOCKS(dY.size())),
        dim3(CAFFE_HIP_NUM_THREADS),
        0,
        context_.hip_stream(), 
        static_cast<const int>(dY.size()),
        dY.data<float>(),
        mask.data<bool>(),
        scale,
        dX->template mutable_data<float>());
    return true;
  }
}

REGISTER_HIP_OPERATOR(Dropout, DropoutOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(DropoutGrad, DropoutGradientOp<float, HIPContext>);
} // namespace caffe2
