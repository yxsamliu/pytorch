#include "hip/hip_runtime.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/utils/math.h"
#include "caffe2/operators/resize_op.h"

namespace caffe2 {

namespace {

__global__ void NearestNeighborKernel(
    const int size,
    const int num_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const float height_scale,
    const float width_scale,
    const float* X,
    float* Y) {
  HIP_1D_KERNEL_LOOP(index, size) {

    int indexTemp = index;
    const int w = indexTemp % output_width;
    indexTemp /= output_width;
    const int h = indexTemp % output_height;
    indexTemp /= output_height;
    const int c = indexTemp % num_channels;
    indexTemp /= num_channels;
    const int n = indexTemp;

    const int in_y = fminf(h / height_scale, input_height - 1);
    const int in_x = fminf(w / width_scale, input_width - 1);
    Y[index] =
        X[((n * num_channels + c) * input_height + in_y) * input_width + in_x];
  }
}

__global__ void NearestNeighborGradientKernel(
    const int size,
    const int num_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const float height_scale,
    const float width_scale,
    const float* dY,
    float* dX) {
  HIP_1D_KERNEL_LOOP(index, size) {
    int indexTemp = index;
    const int x = indexTemp % input_width;
    indexTemp /= input_width;
    const int y = indexTemp % input_height;
    indexTemp /= input_height;
    const int c = indexTemp % num_channels;
    indexTemp /= num_channels;
    const int n = indexTemp;

    const int out_y = fminf(y / height_scale, output_height - 1);
    const int out_x = fminf(x / width_scale, output_width - 1);
    const int out_index =
        ((n * num_channels + c) * output_height + out_y) * output_width + out_x;
#if __HIP_ARCH__ >= 350
    atomicAdd(dX + out_index, __ldg(dY + index));
#else
    atomicAdd(dX + out_index, *(dY + index));
#endif
  }
}

} // namespace

template <>
bool ResizeNearestOp<float, HIPContext>::RunOnDevice() {
  const auto& X = Input(0);
  auto* Y = Output(0);

  const auto inputDims = X.dims();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = X.dim32(0), num_channels = X.dim32(1),
            input_height = X.dim32(2), input_width = X.dim32(3);
  if (InputSize() == 2) {
    const auto& scales = Input(1);
    CAFFE_ENFORCE_EQ(scales.ndim(), 1);
    CAFFE_ENFORCE_EQ(scales.size(), 2);
    float scales_data[2];
    context_.CopyToCPU<float>(2, scales.data<float>(), scales_data);
    height_scale_ = scales_data[0];
    width_scale_ = scales_data[1];
  }
  int output_width = input_width * width_scale_;
  int output_height = input_height * height_scale_;
  Y->Resize(batch_size, num_channels, output_height, output_width);

  const auto size = Y->size();
 hipLaunchKernelGGL( NearestNeighborKernel, 
      dim3(CAFFE_GET_BLOCKS(size)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(size),
      static_cast<const int>(num_channels),
      static_cast<const int>(input_height),
      static_cast<const int>(input_width),
      static_cast<const int>(output_height),
      static_cast<const int>(output_width),
      height_scale_,
      width_scale_,
      X.data<float>(),
      Y->template mutable_data<float>());

  return true;
}

template <>
bool ResizeNearestGradientOp<float, HIPContext>::RunOnDevice() {
  const auto& dY = Input(0);
  const auto& X = Input(1);
  auto* dX = Output(0);

  const auto inputDims = dY.dims();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = dY.dim32(0), num_channels = dY.dim32(1),
            input_height = dY.dim32(2), input_width = dY.dim32(3);
  int output_height = X.dim32(2);
  int output_width = X.dim32(3);
  if (InputSize() == 3) {
    const auto& scales = Input(2);
    CAFFE_ENFORCE_EQ(scales.ndim(), 1);
    CAFFE_ENFORCE_EQ(scales.size(), 2);
    float scales_data[2];
    context_.CopyToCPU<float>(2, scales.data<float>(), scales_data);
    height_scale_ = scales_data[0];
    width_scale_ = scales_data[1];
  }
  dX->Resize(batch_size, num_channels, output_height, output_width);
  math::Set<float, HIPContext>(
      dX->size(), 0.0f, dX->template mutable_data<float>(), &context_);

  const auto size = dY.size();
 hipLaunchKernelGGL( NearestNeighborGradientKernel, 
      dim3(CAFFE_GET_BLOCKS(size)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(size),
      static_cast<const int>(num_channels),
      static_cast<const int>(input_height),
      static_cast<const int>(input_width),
      static_cast<const int>(output_height),
      static_cast<const int>(output_width),
      height_scale_,
      width_scale_,
      dY.data<float>(),
      dX->template mutable_data<float>());

  return true;
}

REGISTER_HIP_OPERATOR(ResizeNearest, ResizeNearestOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    ResizeNearestGradient,
    ResizeNearestGradientOp<float, HIPContext>);
} // namespace caffe2
