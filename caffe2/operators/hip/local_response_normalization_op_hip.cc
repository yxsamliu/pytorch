#include "hip/hip_runtime.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/local_response_normalization_op.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void LRNFillScaleNCHW(const int nthreads, const T* in,
    const int num, const int channels, const int height,
    const int width, const int size, const T alpha_over_size,
    const T bias, T* scale) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    in += offset;
    scale += offset;
    int head = 0;
    int pre_pad = (size - 1) / 2;
    int post_pad = size - pre_pad - 1;
    T accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad) {
      accum_scale += in[head * step] * in[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_scale += in[head * step] * in[head * step];
      scale[(head - post_pad) * step] = bias + accum_scale * alpha_over_size;
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in[head * step] * in[head * step];
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = bias + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_scale -= in[(head - size) * step] * in[(head - size) * step];
      scale[(head - post_pad) * step] = bias + accum_scale * alpha_over_size;
      ++head;
    }
    // recover the pointers for the next loop.
    in -= offset;
    scale -= offset;
  }
}

template <typename T>
__global__ void LRNFillScaleNHWC(const int nthreads, const T* in,
    const int num, const int height, const int width,
    const int channels, const int size, const T alpha_over_size,
    const T bias, T* scale) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    int c = index % channels;
    int pre_pad = (size - 1) / 2;
    scale[index] = 0;
    for (int i = 0; i < size; ++i) {
      int raw_idx = c + i - pre_pad;
      if (raw_idx >= 0 && raw_idx < channels) {
        scale[index] += in[index + i - pre_pad] * in[index + i - pre_pad];
      }
    }
    scale[index] = bias + scale[index] * alpha_over_size;
  }
}

// TODO(Yangqing): check if it would be faster to just put it into the previous
// kernel.
template <typename T>
__global__ void LRNComputeOutput(const int nthreads, const T* in,
    const T* scale, const T negative_beta, T* out) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}

template <typename T>
__global__ void LRNComputeDiffNCHW(const int nthreads, const T* bottom_data,
    const T* top_data, const T* scale, const T* top_diff,
    const int num, const int channels, const int height,
    const int width, const int size, const T negative_beta,
    const T cache_ratio,
    T* bottom_diff) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int n = index / width / height;
    int offset = (n * channels * height + h) * width + w;
    int step = height * width;
    bottom_data += offset;
    top_data += offset;
    scale += offset;
    top_diff += offset;
    bottom_diff += offset;
    int head = 0;
    int pre_pad = size - (size + 1) / 2;
    int post_pad = size - pre_pad - 1;
    T accum_ratio = 0;
    // accumulate values
    while (head < post_pad) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      ++head;
    }
    // until we reach size, nothing needs to be subtracted
    while (head < size) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff[head * step] * top_data[head * step] /
          scale[head * step];
      accum_ratio -= top_diff[(head - size) * step] *
          top_data[(head - size) * step] / scale[(head - size) * step];
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      accum_ratio -= top_diff[(head - size) * step] *
          top_data[(head - size) * step] / scale[(head - size) * step];
      bottom_diff[(head - post_pad) * step] = top_diff[(head - post_pad) * step]
          * pow(scale[(head - post_pad) * step], negative_beta) - cache_ratio *
          bottom_data[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // recover pointer for next iteration.
    bottom_data -= offset;
    top_data -= offset;
    scale -= offset;
    top_diff -= offset;
    bottom_diff -= offset;
  }
}

// This local response normalization gradient does one sum per output location
// and does not use the running trick for 1-d convolution: thus it might not be
// the fastest implementation.
template <typename T>
__global__ void LRNComputeDiffNHWC(const int nthreads, const T* bottom_data,
    const T* top_data, const T* scale, const T* top_diff,
    const int num, const int height, const int width, const int channels,
    const int size, const T negative_beta, const T cache_ratio,
    T* bottom_diff) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local channel offset
    int c = index % channels;
    int pre_pad = size / 2;
    T accum_ratio = 0;
    for (int i = -pre_pad; i < size - pre_pad; ++i) {
      if (c + i >= 0 && c + i < channels) {
        accum_ratio += top_diff[index + i] * top_data[index + i] /
            scale[index + i];
      }
    }
    bottom_diff[index] = top_diff[index] * pow(scale[index], negative_beta) -
                         cache_ratio * bottom_data[index] * accum_ratio;
  }
}
}  // namespace

template<>
bool LRNOp<float, HIPContext>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_EQ(X.ndim(), 4);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);
  const float* Xdata = X.data<float>();
  Y->ResizeLike(X);
  float* Ydata = Y->template mutable_data<float>();
  if (OutputSize() > 1) {
    scale_ = Output(1);
  } else {
    if (!scale_) {
      scale_ = &local_scale_tensor_;
    }
  }
  scale_->ResizeLike(X);
  float* scale_data = scale_->template mutable_data<float>();

  int n_threads = N * H * W;
 hipLaunchKernelGGL( LRNFillScaleNCHW<float>, dim3(CAFFE_GET_BLOCKS(n_threads)), dim3(CAFFE_HIP_NUM_THREADS),
                        0, context_.hip_stream(), 
      static_cast<const int>(n_threads), Xdata, static_cast<const int>(N), static_cast<const int>(C), static_cast<const int>(H), static_cast<const int>(W), static_cast<const int>(size_), alpha_ / static_cast<const int>(size_), bias_, scale_data);
  n_threads = X.size();
 hipLaunchKernelGGL( LRNComputeOutput<float>, dim3(CAFFE_GET_BLOCKS(n_threads)), dim3(CAFFE_HIP_NUM_THREADS),
                            0, context_.hip_stream(), 
      static_cast<const int>(n_threads), Xdata, scale_data, -beta_, Ydata);
  return true;
}

template<>
bool LRNOp<float, HIPContext>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_EQ(X.ndim(), 4);
  const int N = X.dim32(0);
  const int H = X.dim32(1);
  const int W = X.dim32(2);
  const int C = X.dim32(3);
  const float* Xdata = X.data<float>();
  Y->ResizeLike(X);
  float* Ydata = Y->template mutable_data<float>();
  if (OutputSize() > 1) {
    scale_ = Output(1);
  } else {
    if (!scale_) {
      scale_ = &local_scale_tensor_;
    }
  }
  scale_->ResizeLike(X);
  float* scale_data = scale_->template mutable_data<float>();

  int n_threads = X.size();
 hipLaunchKernelGGL( LRNFillScaleNHWC<float>, dim3(CAFFE_GET_BLOCKS(n_threads)), dim3(CAFFE_HIP_NUM_THREADS),
                        0, context_.hip_stream(), 
      static_cast<const int>(n_threads), Xdata, static_cast<const int>(N), static_cast<const int>(H), static_cast<const int>(W), static_cast<const int>(C), static_cast<const int>(size_), alpha_ / static_cast<const int>(size_), bias_, scale_data);
 hipLaunchKernelGGL( LRNComputeOutput<float>, dim3(CAFFE_GET_BLOCKS(n_threads)), dim3(CAFFE_HIP_NUM_THREADS),
                            0, context_.hip_stream(), 
      static_cast<const int>(n_threads), Xdata, scale_data, -beta_, Ydata);
  return true;
}

template <>
bool LRNGradientOp<float, HIPContext>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  DCHECK_EQ(X.ndim(), 4);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);
  // Loosely checking the size, assuming that the shapes will be the same as
  // long as the sizes check out.
  DCHECK_EQ(X.size(), Y.size());
  DCHECK_EQ(X.size(), dY.size());
  dX->ResizeLike(X);

  const float* Xdata = X.data<float>();
  const float* Ydata = Y.data<float>();
  if (!scale_) {
    scale_ = &local_scale_tensor_;
  }
  scale_->ResizeLike(X);
  float* scale_data = scale_->template mutable_data<float>();
  int n_threads = N * H * W;
 hipLaunchKernelGGL( LRNFillScaleNCHW<float>, dim3(CAFFE_GET_BLOCKS(n_threads)), dim3(CAFFE_HIP_NUM_THREADS),
                        0, context_.hip_stream(), 
      static_cast<const int>(n_threads), Xdata, static_cast<const int>(N), static_cast<const int>(C), static_cast<const int>(H), static_cast<const int>(W), static_cast<const int>(size_), alpha_ / static_cast<const int>(size_), bias_, scale_data);
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->template mutable_data<float>();

 hipLaunchKernelGGL( LRNComputeDiffNCHW<float>, dim3(CAFFE_GET_BLOCKS(n_threads)),
                              dim3(CAFFE_HIP_NUM_THREADS),
                              0, context_.hip_stream(), 
      static_cast<const int>(n_threads), Xdata, Ydata, scale_data, dYdata, static_cast<const int>(N), static_cast<const int>(C), static_cast<const int>(H), static_cast<const int>(W), static_cast<const int>(size_), -beta_,
      2.f * alpha_ * beta_ / static_cast<const int>(size_), dXdata);
  return true;
}

template <>
bool LRNGradientOp<float, HIPContext>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);
  auto* dX = Output(0);
  DCHECK_EQ(X.ndim(), 4);
  const int N = X.dim32(0);
  const int H = X.dim32(1);
  const int W = X.dim32(2);
  const int C = X.dim32(3);
  const float* Xdata = X.data<float>();
  // Loosely checking the size, assuming that the shapes will be the same as
  // long as the sizes check out.
  DCHECK_EQ(X.size(), Y.size());
  DCHECK_EQ(X.size(), dY.size());
  dX->ResizeLike(X);
  if (!scale_) {
    scale_ = &local_scale_tensor_;
  }
  scale_->ResizeLike(X);

  float* scale_data = scale_->template mutable_data<float>();
  int n_threads = X.size();
 hipLaunchKernelGGL( LRNFillScaleNHWC<float>, dim3(CAFFE_GET_BLOCKS(n_threads)), dim3(CAFFE_HIP_NUM_THREADS),
                        0, context_.hip_stream(), 
      static_cast<const int>(n_threads), Xdata, static_cast<const int>(N), static_cast<const int>(H), static_cast<const int>(W), static_cast<const int>(C), static_cast<const int>(size_), alpha_ / static_cast<const int>(size_), bias_, scale_data);

 hipLaunchKernelGGL( LRNComputeDiffNHWC<float>
      , dim3(CAFFE_GET_BLOCKS(X.size())),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context_.hip_stream(), 
          static_cast<const int>(X.size()),
          X.data<float>(),
          Y.data<float>(),
          scale_data,
          dY.data<float>(),
          static_cast<const int>(X.dim32(0)),
          static_cast<const int>(X.dim32(1)),
          static_cast<const int>(X.dim32(2)),
          static_cast<const int>(X.dim32(3)),
          static_cast<const int>(size_),
          -beta_,
          2.f * alpha_ * beta_ / static_cast<const int>(size_),
          dX->template mutable_data<float>());
  return true;
}


REGISTER_HIP_OPERATOR(LRN, LRNOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(LRNGradient, LRNGradientOp<float, HIPContext>);

}  // namespace caffe2
