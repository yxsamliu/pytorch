#include "hip/hip_runtime.h"
// TODO(ataei): reduce the apparent redundancy of all the code below.
#include <cfloat>

#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/pool_op.h"

namespace caffe2 {
namespace {
class AveragePool {};
class MaxPool {};
}  // namespace

namespace {
template <typename T>
__global__ void Average1DPoolForwardNCHW(
    const int nthreads,
    const T* bottom_data,
    const int num,
    const int channels,
    const int height,
    const int pooled_height,
    const int kernel_h,
    const int stride_h,
    const int pad_t,
    T* top_data) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int ph = n % pooled_height;
    n /= pooled_height;
    int c = n % channels;
    n /= channels;
    int hstart = ph * stride_h - pad_t;
    int hend = min(hstart + kernel_h, height);
    hstart = max(hstart, 0);
    top_data[index] = 0;
    int bottom_offset = (n * channels + c) * height;
    for (int h = hstart; h < hend; ++h) {
      top_data[index] += bottom_data[bottom_offset + h];
    }
    top_data[index] /= (hend - hstart);
  }
}

template <typename T>
__global__ void Average2DPoolForwardNCHW(
    const int nthreads,
    const T* bottom_data,
    const int num,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    T* top_data) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int pw = n % pooled_width;
    n /= pooled_width;
    int ph = n % pooled_height;
    n /= pooled_height;
    int c = n % channels;
    n /= channels;
    int hstart = ph * stride_h - pad_t;
    int wstart = pw * stride_w - pad_l;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    top_data[index] = 0;
    int bottom_offset = (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        top_data[index] += bottom_data[bottom_offset + h * width + w];
      }
    }
    top_data[index] /= (hend - hstart) * (wend - wstart);
  }
}

template <typename T>
__global__ void Average3DPoolForwardNCHW(
    const int nthreads,
    const T* bottom_data,
    const int num,
    const int channels,
    const int height,
    const int width,
    const int depth,
    const int pooled_height,
    const int pooled_width,
    const int pooled_depth,
    const int kernel_h,
    const int kernel_w,
    const int kernel_d,
    const int stride_h,
    const int stride_w,
    const int stride_d,
    const int pad_t,
    const int pad_l,
    const int pad_f,
    T* top_data) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int pd = n % pooled_depth;
    n /= pooled_depth;
    int pw = n % pooled_width;
    n /= pooled_width;
    int ph = n % pooled_height;
    n /= pooled_height;
    int c = n % channels;
    n /= channels;
    int hstart = ph * stride_h - pad_t;
    int wstart = pw * stride_w - pad_l;
    int dstart = pd * stride_d - pad_f;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    int dend = min(dstart + kernel_d, depth);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    dstart = max(dstart, 0);
    top_data[index] = 0;
    int bottom_offset = (n * channels + c) * height * width * depth;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        for (int d = dstart; d < dend; ++d) {
          const int input_index =
              bottom_offset + h * width * depth + w * depth + d;
          top_data[index] += bottom_data[input_index];
        }
      }
    }
    top_data[index] /= (hend - hstart) * (wend - wstart) * (dend - dstart);
  }
}

template <typename T>
__global__ void Average1DPoolForwardNHWC(
    const int nthreads,
    const T* bottom_data,
    const int num,
    const int height,
    const int channels,
    const int pooled_height,
    const int kernel_h,
    const int stride_h,
    const int pad_t,
    T* top_data) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    int c = index % channels;
    int ph = (index / channels) % pooled_height;
    int n = index / channels / pooled_height;
    int hstart = ph * stride_h - pad_t;
    int hend = min(hstart + kernel_h, height);
    hstart = max(hstart, 0);
    T output = 0;
    int bottom_offset = n * height * channels + c;
    for (int h = hstart; h < hend; ++h) {
      output += bottom_data[bottom_offset + h * channels];
    }
    int pool_size = (hend - hstart);
    top_data[index] = output / pool_size;
  }
}

template <typename T>
__global__ void Average2DPoolForwardNHWC(
    const int nthreads,
    const T* bottom_data,
    const int num,
    const int height,
    const int width,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    T* top_data) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    int c = index % channels;
    int pw = (index / channels) % pooled_width;
    int ph = (index / channels / pooled_width) % pooled_height;
    int n = index / channels / pooled_width / pooled_height;
    int hstart = ph * stride_h - pad_t;
    int wstart = pw * stride_w - pad_l;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    T output = 0;
    int bottom_offset = n * height * width * channels + c;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        output += bottom_data[bottom_offset + (h * width + w) * channels];
      }
    }
    int pool_size = (hend - hstart) * (wend - wstart);
    top_data[index] = output / pool_size;
  }
}

template <typename T>
__global__ void Average3DPoolForwardNHWC(
    const int nthreads,
    const T* bottom_data,
    const int num,
    const int height,
    const int width,
    const int depth,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const int pooled_depth,
    const int kernel_h,
    const int kernel_w,
    const int kernel_d,
    const int stride_h,
    const int stride_w,
    const int stride_d,
    const int pad_t,
    const int pad_l,
    const int pad_f,
    T* top_data) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    int c = index % channels;
    int pd = (index / channels) % pooled_depth;
    int pw = (index / channels / pooled_depth) % pooled_width;
    int ph = (index / channels / pooled_depth / pooled_width) % pooled_height;
    int n = index / channels / pooled_depth / pooled_width / pooled_height;
    int hstart = ph * stride_h - pad_t;
    int wstart = pw * stride_w - pad_l;
    int dstart = pd * stride_d - pad_f;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    int dend = min(dstart + kernel_d, depth);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    dstart = max(dstart, 0);
    T output = 0;
    int bottom_offset = n * height * width * depth * channels + c;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        for (int d = dstart; d < dend; ++d) {
          const int bottom_index =
              bottom_offset + (h * depth * width + w * depth + d) * channels;
          output += bottom_data[bottom_index];
        }
      }
    }
    int pool_size = (hend - hstart) * (wend - wstart) * (dend - dstart);
    top_data[index] = output / pool_size;
  }
}

template <typename T>
__global__ void Ave1DPoolBackwardNCHW(
    const int nthreads,
    const T* const top_diff,
    const int num,
    const int channels,
    const int height,
    const int pooled_height,
    const int kernel_h,
    const int stride_h,
    const int pad_t,
    T* const bottom_diff) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int h = index % height + pad_t;
    const int c = (index / height) % channels;
    const int n = index / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    T gradient = 0;
    const T* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height;
    for (int ph = phstart; ph < phend; ++ph) {
      // figure out the pooling size
      int hstart = ph * stride_h - pad_t;
      int hend = min(hstart + kernel_h, height);
      hstart = max(hstart, 0);
      int pool_size = (hend - hstart);
      gradient += top_diff_slice[ph] / pool_size;
    }
    bottom_diff[index] = gradient;
  }
}

template <typename T>
__global__ void Ave2DPoolBackwardNCHW(
    const int nthreads,
    const T* const top_diff,
    const int num,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    T* const bottom_diff) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_l;
    const int h = (index / width) % height + pad_t;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    T gradient = 0;
    const T* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_t;
        int wstart = pw * stride_w - pad_l;
        int hend = min(hstart + kernel_h, height);
        int wend = min(wstart + kernel_w, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename T>
__global__ void Ave3DPoolBackwardNCHW(
    const int nthreads,
    const T* const top_diff,
    const int num,
    const int channels,
    const int height,
    const int width,
    const int depth,
    const int pooled_height,
    const int pooled_width,
    const int pooled_depth,
    const int kernel_h,
    const int kernel_w,
    const int kernel_d,
    const int stride_h,
    const int stride_w,
    const int stride_d,
    const int pad_t,
    const int pad_l,
    const int pad_f,
    T* const bottom_diff) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int d = index % depth + pad_f;
    const int w = (index / depth) % width + pad_l;
    const int h = (index / depth / width) % height + pad_t;
    const int c = (index / depth / width / height) % channels;
    const int n = index / depth / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    const int pdstart = (d < kernel_d) ? 0 : (d - kernel_d) / stride_d + 1;
    const int pdend = min(d / stride_d + 1, pooled_depth);
    T gradient = 0;
    const T* const top_diff_slice = top_diff +
        (n * channels + c) * pooled_height * pooled_width * pooled_depth;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        for (int pd = pdstart; pd < pdend; ++pd) {
          // figure out the pooling size
          int hstart = ph * stride_h - pad_t;
          int wstart = pw * stride_w - pad_l;
          int dstart = pd * stride_d - pad_f;
          int hend = min(hstart + kernel_h, height);
          int wend = min(wstart + kernel_w, width);
          int dend = min(dstart + kernel_d, depth);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          dstart = max(dstart, 0);
          int pool_size = (hend - hstart) * (wend - wstart) * (dend - dstart);
          const int pooled_index =
              ph * pooled_depth * pooled_width + pooled_depth * pw + pd;
          gradient += top_diff_slice[pooled_index] / pool_size;
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename T>
__global__ void Ave1DPoolBackwardNHWC(
    const int nthreads,
    const T* const top_diff,
    const int num,
    const int height,
    const int channels,
    const int pooled_height,
    const int kernel_h,
    const int stride_h,
    const int pad_t,
    T* const bottom_diff) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int c = index % channels;
    const int h = (index / channels) % height + pad_t;
    const int n = index / channels / height;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    T gradient = 0;
    const T* const top_diff_slice = top_diff + n * pooled_height * channels + c;
    for (int ph = phstart; ph < phend; ++ph) {
      // figure out the pooling size
      int hstart = ph * stride_h - pad_t;
      int hend = min(hstart + kernel_h, height);
      hstart = max(hstart, 0);
      int pool_size = (hend - hstart);
      gradient += top_diff_slice[ph * channels] / pool_size;
    }
    bottom_diff[index] = gradient;
  }
}

template <typename T>
__global__ void Ave2DPoolBackwardNHWC(
    const int nthreads,
    const T* const top_diff,
    const int num,
    const int height,
    const int width,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    T* const bottom_diff) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int c = index % channels;
    const int w = index / channels % width + pad_l;
    const int h = (index / channels / width) % height + pad_t;
    const int n = index / channels / width / height;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    T gradient = 0;
    const T* const top_diff_slice =
        top_diff + n * pooled_height * pooled_width * channels + c;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_t;
        int wstart = pw * stride_w - pad_l;
        int hend = min(hstart + kernel_h, height);
        int wend = min(wstart + kernel_w, width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient +=
            top_diff_slice[(ph * pooled_width + pw) * channels] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename T>
__global__ void Ave3DPoolBackwardNHWC(
    const int nthreads,
    const T* const top_diff,
    const int num,
    const int height,
    const int width,
    const int depth,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const int pooled_depth,
    const int kernel_h,
    const int kernel_w,
    const int kernel_d,
    const int stride_h,
    const int stride_w,
    const int stride_d,
    const int pad_t,
    const int pad_l,
    const int pad_f,
    T* const bottom_diff) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int c = index % channels;
    const int d = index / channels % depth + pad_f;
    const int w = (index / channels / depth) % width + pad_l;
    const int h = (index / channels / depth / width) % height + pad_t;
    const int n = index / channels / depth / width / height;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    const int pdstart = (d < kernel_d) ? 0 : (d - kernel_d) / stride_d + 1;
    const int pdend = min(d / stride_d + 1, pooled_depth);
    T gradient = 0;
    const T* const top_diff_slice = top_diff +
        n * pooled_height * pooled_width * pooled_depth * channels + c;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        for (int pd = pdstart; pd < pdend; ++pd) {
          // figure out the pooling size
          int hstart = ph * stride_h - pad_t;
          int wstart = pw * stride_w - pad_l;
          int dstart = pd * stride_d - pad_f;
          int hend = min(hstart + kernel_h, height);
          int wend = min(wstart + kernel_w, width);
          int dend = min(dstart + kernel_d, depth);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          dstart = max(dstart, 0);
          int pool_size = (hend - hstart) * (wend - wstart) * (dend - dstart);
          const int pooled_index =
              (ph * pooled_depth * pooled_width + pw * pooled_depth + pd) *
              channels;
          gradient += top_diff_slice[pooled_index] / pool_size;
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

}  // namespace

template <>
bool PoolOp<float, HIPContext, AveragePool>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase<HIPContext>::SetOutputSize(X, Y, X.dim32(1));
  int output_size = Y->size();
  switch (kernel_.size()) {
    case 1:
     hipLaunchKernelGGL( Average1DPoolForwardNCHW<float>
          , dim3(CAFFE_GET_BLOCKS(output_size)),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(output_size),
              X.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(Y->dim32(2)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(stride_h()),
              static_cast<const int>(pad_t()),
              Y->template mutable_data<float>());
      break;
    case 2:
     hipLaunchKernelGGL( Average2DPoolForwardNCHW<float>
          , dim3(CAFFE_GET_BLOCKS(output_size)),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(output_size),
              X.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(X.dim32(3)),
              static_cast<const int>(Y->dim32(2)),
              static_cast<const int>(Y->dim32(3)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(kernel_w()),
              static_cast<const int>(stride_h()),
              static_cast<const int>(stride_w()),
              static_cast<const int>(pad_t()),
              static_cast<const int>(pad_l()),
              Y->template mutable_data<float>());
      break;
    case 3:
     hipLaunchKernelGGL( Average3DPoolForwardNCHW<float>
          , dim3(CAFFE_GET_BLOCKS(output_size)),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(output_size),
              X.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(X.dim32(3)),
              static_cast<const int>(X.dim32(4)),
              static_cast<const int>(Y->dim32(2)),
              static_cast<const int>(Y->dim32(3)),
              static_cast<const int>(Y->dim32(4)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(kernel_w()),
              static_cast<const int>(kernel_[2]),
              static_cast<const int>(stride_h()),
              static_cast<const int>(stride_w()),
              static_cast<const int>(stride_[2]),
              static_cast<const int>(pad_t()),
              static_cast<const int>(pad_l()),
              static_cast<const int>(pads_[2]),
              Y->template mutable_data<float>());
      break;
    default:
      CAFFE_THROW("Unsupported pooling size : ", kernel_.size());
  }
  return true;
}

template <>
bool PoolOp<float, HIPContext, AveragePool>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase<HIPContext>::SetOutputSize(X, Y, X.dim32(X.ndim() - 1));
  int output_size = Y->size();
  switch (kernel_.size()) {
    case 1:
     hipLaunchKernelGGL( Average1DPoolForwardNHWC<float>
          , dim3(CAFFE_GET_BLOCKS(output_size)),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(output_size),
              X.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(Y->dim32(1)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(stride_h()),
              static_cast<const int>(pad_t()),
              Y->template mutable_data<float>());
      break;
    case 2:
     hipLaunchKernelGGL( Average2DPoolForwardNHWC<float>
          , dim3(CAFFE_GET_BLOCKS(output_size)),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(output_size),
              X.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(X.dim32(3)),
              static_cast<const int>(Y->dim32(1)),
              static_cast<const int>(Y->dim32(2)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(kernel_w()),
              static_cast<const int>(stride_h()),
              static_cast<const int>(stride_w()),
              static_cast<const int>(pad_t()),
              static_cast<const int>(pad_l()),
              Y->template mutable_data<float>());
      break;
    case 3:
     hipLaunchKernelGGL( Average3DPoolForwardNHWC<float>
          , dim3(CAFFE_GET_BLOCKS(output_size)),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(output_size),
              X.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(X.dim32(3)),
              static_cast<const int>(X.dim32(4)),
              static_cast<const int>(Y->dim32(1)),
              static_cast<const int>(Y->dim32(2)),
              static_cast<const int>(Y->dim32(3)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(kernel_w()),
              static_cast<const int>(kernel_[2]),
              static_cast<const int>(stride_h()),
              static_cast<const int>(stride_w()),
              static_cast<const int>(stride_[2]),
              static_cast<const int>(pad_t()),
              static_cast<const int>(pad_l()),
              static_cast<const int>(pads_[2]),
              Y->template mutable_data<float>());
      break;
    default:
      CAFFE_THROW("Unsupported pooling size : ", kernel_.size());
  }
  return true;
}

template <>
bool PoolGradientOp<float, HIPContext, AveragePool>::
    RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto& dY = Input(2);
  CAFFE_ENFORCE_EQ(dY.dim32(1), X.dim32(1));
  auto* dX = Output(0);
  dX->ResizeLike(X);
  vector<int> dims(X.dims().begin() + 2, X.dims().end());
  ConvPoolOpBase<HIPContext>::ComputePads(dims);
  switch (kernel_.size()) {
    case 1:
     hipLaunchKernelGGL( Ave1DPoolBackwardNCHW<float>
          , dim3(CAFFE_GET_BLOCKS(X.size())),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(X.size()),
              dY.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(dY.dim32(2)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(stride_h()),
              static_cast<const int>(pad_t()),
              dX->template mutable_data<float>());
      break;
    case 2:
     hipLaunchKernelGGL( Ave2DPoolBackwardNCHW<float>
          , dim3(CAFFE_GET_BLOCKS(X.size())),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(X.size()),
              dY.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(X.dim32(3)),
              static_cast<const int>(dY.dim32(2)),
              static_cast<const int>(dY.dim32(3)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(kernel_w()),
              static_cast<const int>(stride_h()),
              static_cast<const int>(stride_w()),
              static_cast<const int>(pad_t()),
              static_cast<const int>(pad_l()),
              dX->template mutable_data<float>());
      break;
    case 3:
     hipLaunchKernelGGL( Ave3DPoolBackwardNCHW<float>
          , dim3(CAFFE_GET_BLOCKS(X.size())),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(X.size()),
              dY.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(X.dim32(3)),
              static_cast<const int>(X.dim32(4)),
              static_cast<const int>(dY.dim32(2)),
              static_cast<const int>(dY.dim32(3)),
              static_cast<const int>(dY.dim32(4)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(kernel_w()),
              static_cast<const int>(kernel_[2]),
              static_cast<const int>(stride_h()),
              static_cast<const int>(stride_w()),
              static_cast<const int>(stride_[2]),
              static_cast<const int>(pad_t()),
              static_cast<const int>(pad_l()),
              static_cast<const int>(pads_[2]),
              dX->template mutable_data<float>());
      break;
    default:
      CAFFE_THROW("Unsupported pooling size : ", kernel_.size());
  }
  return true;
}

template <>
bool PoolGradientOp<float, HIPContext, AveragePool>::
    RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto& dY = Input(2);
  CAFFE_ENFORCE_EQ(X.ndim(), dY.ndim());
  CAFFE_ENFORCE_EQ(X.dim32(X.ndim() - 1), dY.dim32(dY.ndim() - 1));
  auto* dX = Output(0);
  dX->ResizeLike(X);
  vector<int> dims(X.dims().begin() + 1, X.dims().end() - 1);
  ConvPoolOpBase<HIPContext>::ComputePads(dims);
  switch (kernel_.size()) {
    case 1:
     hipLaunchKernelGGL( Ave1DPoolBackwardNHWC<float>
          , dim3(CAFFE_GET_BLOCKS(X.size())),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(X.size()),
              dY.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(dY.dim32(1)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(stride_h()),
              static_cast<const int>(pad_t()),
              dX->template mutable_data<float>());
      break;
    case 2:
     hipLaunchKernelGGL( Ave2DPoolBackwardNHWC<float>
          , dim3(CAFFE_GET_BLOCKS(X.size())),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(X.size()),
              dY.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(X.dim32(3)),
              static_cast<const int>(dY.dim32(1)),
              static_cast<const int>(dY.dim32(2)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(kernel_w()),
              static_cast<const int>(stride_h()),
              static_cast<const int>(stride_w()),
              static_cast<const int>(pad_t()),
              static_cast<const int>(pad_l()),
              dX->template mutable_data<float>());
      break;
    case 3:
     hipLaunchKernelGGL( Ave3DPoolBackwardNHWC<float>
          , dim3(CAFFE_GET_BLOCKS(X.size())),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(X.size()),
              dY.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(X.dim32(3)),
              static_cast<const int>(X.dim32(4)),
              static_cast<const int>(dY.dim32(1)),
              static_cast<const int>(dY.dim32(2)),
              static_cast<const int>(dY.dim32(3)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(kernel_w()),
              static_cast<const int>(kernel_[2]),
              static_cast<const int>(stride_h()),
              static_cast<const int>(stride_w()),
              static_cast<const int>(stride_[2]),
              static_cast<const int>(pad_t()),
              static_cast<const int>(pad_l()),
              static_cast<const int>(pads_[2]),
              dX->template mutable_data<float>());
      break;
    default:
      CAFFE_THROW("Unsupported pooling size : ", kernel_.size());
  }
  return true;
}


namespace {

template <typename T>
__global__ void MaxPool1DForwardNCHW(
    const int nthreads,
    const T* bottom_data,
    const int channels,
    const int height,
    const int pooled_height,
    const int kernel_h,
    const int stride_h,
    const int pad_t,
    T* top_data) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    int ph = index % pooled_height;
    int c = (index / pooled_height) % channels;
    int n = index / pooled_height / channels;
    int hstart = ph * stride_h - pad_t;
    int hend = min(hstart + kernel_h, height);
    hstart = max(hstart, 0);
    T maxval = -FLT_MAX;
    const T* bdata_offset = bottom_data + n * channels * height;
    for (int h = hstart; h < hend; ++h) {
      int idx = c * height + h;
      if (bdata_offset[idx] > maxval) {
        maxval = bdata_offset[idx];
      }
    }
    top_data[index] = maxval;
  }
}

template <typename T>
__global__ void MaxPool2DForwardNCHW(
    const int nthreads,
    const T* bottom_data,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    T* top_data) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_t;
    int wstart = pw * stride_w - pad_l;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    T maxval = -FLT_MAX;
    const T* bdata_offset = bottom_data + n * channels * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int idx = c * height * width + h * width + w;
        if (bdata_offset[idx] > maxval) {
          maxval = bdata_offset[idx];
        }
      }
    }
    top_data[index] = maxval;
  }
}

template <typename T>
__global__ void MaxPool3DForwardNCHW(
    const int nthreads,
    const T* bottom_data,
    const int channels,
    const int height,
    const int width,
    const int depth,
    const int pooled_height,
    const int pooled_width,
    const int pooled_depth,
    const int kernel_h,
    const int kernel_w,
    const int kernel_d,
    const int stride_h,
    const int stride_w,
    const int stride_d,
    const int pad_t,
    const int pad_l,
    const int pad_f,
    T* top_data) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    int pd = index % pooled_depth;
    int pw = (index / pooled_depth) % pooled_width;
    int ph = (index / pooled_depth / pooled_width) % pooled_height;
    int c = (index / pooled_depth / pooled_width / pooled_height) % channels;
    int n = index / pooled_depth / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_t;
    int wstart = pw * stride_w - pad_l;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    int dstart = pd * stride_d - pad_f;
    int dend = min(dstart + kernel_d, depth);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    dstart = max(dstart, 0);
    T maxval = -FLT_MAX;
    const T* bdata_offset = bottom_data + n * channels * height * width * depth;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        for (int d = dstart; d < dend; ++d) {
          int idx = ((c * height + h) * width + w) * depth + d;
          if (bdata_offset[idx] > maxval) {
            maxval = bdata_offset[idx];
          }
        }
      }
    }
    top_data[index] = maxval;
  }
}

template <typename T>
__global__ void MaxPool1DForwardNHWC(
    const int nthreads,
    const T* bottom_data,
    const int height,
    const int channels,
    const int pooled_height,
    const int kernel_h,
    const int stride_h,
    const int pad_t,
    T* top_data) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int hstart = (n % pooled_height) * stride_h - pad_t;
    n /= pooled_height;
    int hend = min(hstart + kernel_h, height);
    hstart = max(hstart, 0);
    T maxval = -FLT_MAX;
    const T* bdata_offset = bottom_data + n * height * channels;
    for (int h = hstart; h < hend; ++h) {
      int idx = h * channels + c;
      if (bdata_offset[idx] > maxval) {
        maxval = bdata_offset[idx];
      }
    }
    top_data[index] = maxval;
  }
}

template <typename T>
__global__ void MaxPool2DForwardNHWC(
    const int nthreads,
    const T* bottom_data,
    const int height,
    const int width,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    T* top_data) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int wstart = (n % pooled_width) * stride_w - pad_l;
    n /= pooled_width;
    int hstart = (n % pooled_height) * stride_h - pad_t;
    n /= pooled_height;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    T maxval = -FLT_MAX;
    const T* bdata_offset = bottom_data + n * height * width * channels;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int idx = (h * width + w) * channels + c;
        if (bdata_offset[idx] > maxval) {
          maxval = bdata_offset[idx];
        }
      }
    }
    top_data[index] = maxval;
  }
}

template <typename T>
__global__ void MaxPool3DForwardNHWC(
    const int nthreads,
    const T* bottom_data,
    const int height,
    const int width,
    const int depth,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const int pooled_depth,
    const int kernel_h,
    const int kernel_w,
    const int kernel_d,
    const int stride_h,
    const int stride_w,
    const int stride_d,
    const int pad_t,
    const int pad_l,
    const int pad_f,
    T* top_data) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int dstart = (n % pooled_depth) * stride_d - pad_f;
    n /= pooled_depth;
    int wstart = (n % pooled_width) * stride_w - pad_l;
    n /= pooled_width;
    int hstart = (n % pooled_height) * stride_h - pad_t;
    n /= pooled_height;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    int dend = min(dstart + kernel_d, depth);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    dstart = max(dstart, 0);
    T maxval = -FLT_MAX;
    const T* bdata_offset = bottom_data + n * height * width * depth * channels;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        for (int d = dstart; d < dend; ++d) {
          int idx = ((h * width + w) * depth + d) * channels + c;
          if (bdata_offset[idx] > maxval) {
            maxval = bdata_offset[idx];
          }
        }
      }
    }
    top_data[index] = maxval;
  }
}

template <typename T>
__global__ void MaxPool1DBackwardNCHW(
    const int nthreads,
    const T* const bottom_data,
    const T* const top_data,
    const T* const top_diff,
    const int num,
    const int channels,
    const int height,
    const int pooled_height,
    const int kernel_h,
    const int stride_h,
    const int pad_t,
    T* const bottom_diff) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int h = index % height + pad_t;
    const int c = (index / height) % channels;
    const int n = index / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int top_offset = (n * channels + c) * pooled_height;
    bottom_diff[index] = 0;
    for (int ph = phstart; ph < phend; ++ph) {
      int top_local_offset = top_offset + ph;
      if (bottom_data[index] == top_data[top_local_offset]) {
        bottom_diff[index] += top_diff[top_local_offset];
      }
    }
  }
}

template <typename T>
__global__ void MaxPool2DBackwardNCHW(
    const int nthreads,
    const T* const bottom_data,
    const T* const top_data,
    const T* const top_diff,
    const int num,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    T* const bottom_diff) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_l;
    const int h = (index / width) % height + pad_t;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    const int top_offset = (n * channels + c) * pooled_height * pooled_width;
    bottom_diff[index] = 0;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int top_local_offset = top_offset + ph * pooled_width + pw;
        if (bottom_data[index] == top_data[top_local_offset]) {
          bottom_diff[index] += top_diff[top_local_offset];
        }
      }
    }
  }
}

template <typename T>
__global__ void MaxPool3DBackwardNCHW(
    const int nthreads,
    const T* const bottom_data,
    const T* const top_data,
    const T* const top_diff,
    const int num,
    const int channels,
    const int height,
    const int width,
    const int depth,
    const int pooled_height,
    const int pooled_width,
    const int pooled_depth,
    const int kernel_h,
    const int kernel_w,
    const int kernel_d,
    const int stride_h,
    const int stride_w,
    const int stride_d,
    const int pad_t,
    const int pad_l,
    const int pad_f,
    T* const bottom_diff) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int d = index % depth + pad_f;
    const int w = (index / depth) % width + pad_l;
    const int h = (index / depth / width) % height + pad_t;
    const int c = (index / depth / width / height) % channels;
    const int n = index / depth / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    const int pdstart = (d < kernel_d) ? 0 : (d - kernel_d) / stride_d + 1;
    const int pdend = min(d / stride_d + 1, pooled_depth);
    const int top_offset =
        (n * channels + c) * pooled_height * pooled_width * pooled_depth;
    bottom_diff[index] = 0;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        for (int pd = pdstart; pd < pdend; ++pd) {
          int top_local_offset =
              top_offset + (ph * pooled_width + pw) * pooled_depth + pd;
          if (bottom_data[index] == top_data[top_local_offset]) {
            bottom_diff[index] += top_diff[top_local_offset];
          }
        }
      }
    }
  }
}

template <typename T>
__global__ void MaxPool1DBackwardNHWC(
    const int nthreads,
    const T* const bottom_data,
    const T* const top_data,
    const T* const top_diff,
    const int num,
    const int height,
    const int channels,
    const int pooled_height,
    const int kernel_h,
    const int stride_h,
    const int pad_t,
    T* const bottom_diff) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int c = index % channels;
    const int h = (index / channels) % height + pad_t;
    const int n = index / channels / height;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int top_offset = n * pooled_height * channels + c;
    bottom_diff[index] = 0;
    for (int ph = phstart; ph < phend; ++ph) {
      int top_local_offset = top_offset + ph * channels;
      if (bottom_data[index] == top_data[top_local_offset]) {
        bottom_diff[index] += top_diff[top_local_offset];
      }
    }
  }
}

template <typename T>
__global__ void MaxPool2DBackwardNHWC(
    const int nthreads,
    const T* const bottom_data,
    const T* const top_data,
    const T* const top_diff,
    const int num,
    const int height,
    const int width,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_t,
    const int pad_l,
    T* const bottom_diff) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int c = index % channels;
    const int w = index / channels % width + pad_l;
    const int h = (index / channels / width) % height + pad_t;
    const int n = index / channels / width / height;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    const int top_offset =
        n * pooled_height * pooled_width * channels + c;
    bottom_diff[index] = 0;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int top_local_offset = top_offset + (ph * pooled_width + pw) * channels;
        if (bottom_data[index] == top_data[top_local_offset]) {
          bottom_diff[index] += top_diff[top_local_offset];
        }
      }
    }
  }
}

template <typename T>
__global__ void MaxPool3DBackwardNHWC(
    const int nthreads,
    const T* const bottom_data,
    const T* const top_data,
    const T* const top_diff,
    const int num,
    const int height,
    const int width,
    const int depth,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const int pooled_depth,
    const int kernel_h,
    const int kernel_w,
    const int kernel_d,
    const int stride_h,
    const int stride_w,
    const int stride_d,
    const int pad_t,
    const int pad_l,
    const int pad_f,
    T* const bottom_diff) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int c = index % channels;
    const int d = index / channels % depth + pad_f;
    const int w = (index / depth / channels) % width + pad_l;
    const int h = (index / channels / depth / width) % height + pad_t;
    const int n = index / channels / depth / width / height;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    const int pdstart = (d < kernel_d) ? 0 : (d - kernel_d) / stride_d + 1;
    const int pdend = min(d / stride_d + 1, pooled_depth);
    const int top_offset =
        n * pooled_height * pooled_width * pooled_depth * channels + c;
    bottom_diff[index] = 0;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        for (int pd = pdstart; pd < pdend; ++pd) {
          int top_local_offset = top_offset +
              ((ph * pooled_width + pw) * pooled_depth + d) * channels;
          if (bottom_data[index] == top_data[top_local_offset]) {
            bottom_diff[index] += top_diff[top_local_offset];
          }
        }
      }
    }
  }
}
}  // namespace

template <>
bool PoolOp<float, HIPContext, MaxPool>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase<HIPContext>::SetOutputSize(X, Y, X.dim32(1));
  int output_size = Y->size();
  switch (kernel_.size()) {
    case 1:
     hipLaunchKernelGGL( MaxPool1DForwardNCHW<float>
          , dim3(CAFFE_GET_BLOCKS(output_size)),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(output_size),
              X.data<float>(),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(Y->dim32(2)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(stride_h()),
              static_cast<const int>(pad_t()),
              Y->template mutable_data<float>());
      break;
    case 2:
     hipLaunchKernelGGL( MaxPool2DForwardNCHW<float>
          , dim3(CAFFE_GET_BLOCKS(output_size)),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(output_size),
              X.data<float>(),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(X.dim32(3)),
              static_cast<const int>(Y->dim32(2)),
              static_cast<const int>(Y->dim32(3)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(kernel_w()),
              static_cast<const int>(stride_h()),
              static_cast<const int>(stride_w()),
              static_cast<const int>(pad_t()),
              static_cast<const int>(pad_l()),
              Y->template mutable_data<float>());
      break;
    case 3:
     hipLaunchKernelGGL( MaxPool3DForwardNCHW<float>
          , dim3(CAFFE_GET_BLOCKS(output_size)),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(output_size),
              X.data<float>(),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(X.dim32(3)),
              static_cast<const int>(X.dim32(4)),
              static_cast<const int>(Y->dim32(2)),
              static_cast<const int>(Y->dim32(3)),
              static_cast<const int>(Y->dim32(4)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(kernel_w()),
              static_cast<const int>(kernel_[2]),
              static_cast<const int>(stride_h()),
              static_cast<const int>(stride_w()),
              static_cast<const int>(stride_[2]),
              static_cast<const int>(pad_t()),
              static_cast<const int>(pad_l()),
              static_cast<const int>(pads_[2]),
              Y->template mutable_data<float>());
      break;
    default:
      CAFFE_THROW("Unsupported pooling size : ", kernel_.size());
  }
  return true;
}

template <>
bool PoolOp<float, HIPContext, MaxPool>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase<HIPContext>::SetOutputSize(X, Y, X.dim32(X.ndim() - 1));
  int output_size = Y->size();
  switch (kernel_.size()) {
    case 1:
     hipLaunchKernelGGL( MaxPool1DForwardNHWC<float>
          , dim3(CAFFE_GET_BLOCKS(output_size)),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(output_size),
              X.data<float>(),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(Y->dim32(1)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(stride_h()),
              static_cast<const int>(pad_t()),
              Y->template mutable_data<float>());
      break;
    case 2:
     hipLaunchKernelGGL( MaxPool2DForwardNHWC<float>
          , dim3(CAFFE_GET_BLOCKS(output_size)),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(output_size),
              X.data<float>(),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(X.dim32(3)),
              static_cast<const int>(Y->dim32(1)),
              static_cast<const int>(Y->dim32(2)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(kernel_w()),
              static_cast<const int>(stride_h()),
              static_cast<const int>(stride_w()),
              static_cast<const int>(pad_t()),
              static_cast<const int>(pad_l()),
              Y->template mutable_data<float>());
      break;
    case 3:
     hipLaunchKernelGGL( MaxPool3DForwardNHWC<float>
          , dim3(CAFFE_GET_BLOCKS(output_size)),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(output_size),
              X.data<float>(),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(X.dim32(3)),
              static_cast<const int>(X.dim32(4)),
              static_cast<const int>(Y->dim32(1)),
              static_cast<const int>(Y->dim32(2)),
              static_cast<const int>(Y->dim32(3)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(kernel_w()),
              static_cast<const int>(kernel_[2]),
              static_cast<const int>(stride_h()),
              static_cast<const int>(stride_w()),
              static_cast<const int>(stride_[2]),
              static_cast<const int>(pad_t()),
              static_cast<const int>(pad_l()),
              static_cast<const int>(pads_[2]),
              Y->template mutable_data<float>());
      break;
    default:
      CAFFE_THROW("Unsupported pooling size : ", kernel_.size());
  }
  return true;
}

template <>
bool PoolGradientOp<float, HIPContext, MaxPool>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);
  CAFFE_ENFORCE_EQ(dY.ndim(), X.ndim());
  auto* dX = Output(0);
  dX->ResizeLike(X);
  vector<int> dims(X.dims().begin() + 2, X.dims().end());
  ConvPoolOpBase<HIPContext>::ComputePads(dims);
  switch (kernel_.size()) {
    case 1:
     hipLaunchKernelGGL( MaxPool1DBackwardNCHW<float>
          , dim3(CAFFE_GET_BLOCKS(X.size())),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(X.size()),
              X.data<float>(),
              Y.data<float>(),
              dY.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(dY.dim32(2)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(stride_h()),
              static_cast<const int>(pad_t()),
              dX->template mutable_data<float>());
      break;
    case 2:
     hipLaunchKernelGGL( MaxPool2DBackwardNCHW<float>
          , dim3(CAFFE_GET_BLOCKS(X.size())),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(X.size()),
              X.data<float>(),
              Y.data<float>(),
              dY.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(X.dim32(3)),
              static_cast<const int>(dY.dim32(2)),
              static_cast<const int>(dY.dim32(3)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(kernel_w()),
              static_cast<const int>(stride_h()),
              static_cast<const int>(stride_w()),
              static_cast<const int>(pad_t()),
              static_cast<const int>(pad_l()),
              dX->template mutable_data<float>());
      break;
    case 3:
     hipLaunchKernelGGL( MaxPool3DBackwardNCHW<float>
          , dim3(CAFFE_GET_BLOCKS(X.size())),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(X.size()),
              X.data<float>(),
              Y.data<float>(),
              dY.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(X.dim32(3)),
              static_cast<const int>(X.dim32(4)),
              static_cast<const int>(dY.dim32(2)),
              static_cast<const int>(dY.dim32(3)),
              static_cast<const int>(dY.dim32(4)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(kernel_w()),
              static_cast<const int>(kernel_[2]),
              static_cast<const int>(stride_h()),
              static_cast<const int>(stride_w()),
              static_cast<const int>(stride_[2]),
              static_cast<const int>(pad_t()),
              static_cast<const int>(pad_l()),
              static_cast<const int>(pads_[2]),
              dX->template mutable_data<float>());
      break;
    default:
      CAFFE_THROW("Unsupported pooling size : ", kernel_.size());
  }
  return true;
}

template <>
bool PoolGradientOp<float, HIPContext, MaxPool>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dY = Input(2);
  CAFFE_ENFORCE_EQ(dY.ndim(), X.ndim());
  auto* dX = Output(0);
  dX->ResizeLike(X);
  vector<int> dims(X.dims().begin() + 1, X.dims().end() - 1);
  ConvPoolOpBase<HIPContext>::ComputePads(dims);
  switch (kernel_.size()) {
    case 1:
     hipLaunchKernelGGL( MaxPool1DBackwardNHWC<float>
          , dim3(CAFFE_GET_BLOCKS(X.size())),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(X.size()),
              X.data<float>(),
              Y.data<float>(),
              dY.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(dY.dim32(1)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(stride_h()),
              static_cast<const int>(pad_t()),
              dX->template mutable_data<float>());
      break;
    case 2:
     hipLaunchKernelGGL( MaxPool2DBackwardNHWC<float>
          , dim3(CAFFE_GET_BLOCKS(X.size())),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(X.size()),
              X.data<float>(),
              Y.data<float>(),
              dY.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(X.dim32(3)),
              static_cast<const int>(dY.dim32(1)),
              static_cast<const int>(dY.dim32(2)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(kernel_w()),
              static_cast<const int>(stride_h()),
              static_cast<const int>(stride_w()),
              static_cast<const int>(pad_t()),
              static_cast<const int>(pad_l()),
              dX->template mutable_data<float>());
      break;
    case 3:
     hipLaunchKernelGGL( MaxPool3DBackwardNHWC<float>
          , dim3(CAFFE_GET_BLOCKS(X.size())),
             dim3(CAFFE_HIP_NUM_THREADS),
             0,
             context_.hip_stream(), 
              static_cast<const int>(X.size()),
              X.data<float>(),
              Y.data<float>(),
              dY.data<float>(),
              static_cast<const int>(X.dim32(0)),
              static_cast<const int>(X.dim32(1)),
              static_cast<const int>(X.dim32(2)),
              static_cast<const int>(X.dim32(3)),
              static_cast<const int>(X.dim32(4)),
              static_cast<const int>(dY.dim32(1)),
              static_cast<const int>(dY.dim32(2)),
              static_cast<const int>(dY.dim32(3)),
              static_cast<const int>(kernel_h()),
              static_cast<const int>(kernel_w()),
              static_cast<const int>(kernel_[2]),
              static_cast<const int>(stride_h()),
              static_cast<const int>(stride_w()),
              static_cast<const int>(stride_[2]),
              static_cast<const int>(pad_t()),
              static_cast<const int>(pad_l()),
              static_cast<const int>(pads_[2]),
              dX->template mutable_data<float>());
      break;
    default:
      CAFFE_THROW("Unsupported pooling size : ", kernel_.size());
  }
  return true;
}

REGISTER_HIP_OPERATOR(AveragePool, PoolOp<float, HIPContext, AveragePool>);
REGISTER_HIP_OPERATOR(AveragePoolGradient,
                       PoolGradientOp<float, HIPContext, AveragePool>);

REGISTER_HIP_OPERATOR(AveragePool1D, PoolOp<float, HIPContext, AveragePool>);
REGISTER_HIP_OPERATOR(
    AveragePool1DGradient,
    PoolGradientOp<float, HIPContext, AveragePool>);

REGISTER_HIP_OPERATOR(AveragePool2D, PoolOp<float, HIPContext, AveragePool>);
REGISTER_HIP_OPERATOR(
    AveragePool2DGradient,
    PoolGradientOp<float, HIPContext, AveragePool>);

REGISTER_HIP_OPERATOR(AveragePool3D, PoolOp<float, HIPContext, AveragePool>);
REGISTER_HIP_OPERATOR(
    AveragePool3DGradient,
    PoolGradientOp<float, HIPContext, AveragePool>);

REGISTER_HIP_OPERATOR(MaxPool, PoolOp<float, HIPContext, MaxPool>);
REGISTER_HIP_OPERATOR(MaxPoolGradient,
                       PoolGradientOp<float, HIPContext, MaxPool>);

REGISTER_HIP_OPERATOR(MaxPool1D, PoolOp<float, HIPContext, MaxPool>);
REGISTER_HIP_OPERATOR(
    MaxPool1DGradient,
    PoolGradientOp<float, HIPContext, MaxPool>);

REGISTER_HIP_OPERATOR(MaxPool2D, PoolOp<float, HIPContext, MaxPool>);
REGISTER_HIP_OPERATOR(
    MaxPool2DGradient,
    PoolGradientOp<float, HIPContext, MaxPool>);

REGISTER_HIP_OPERATOR(MaxPool3D, PoolOp<float, HIPContext, MaxPool>);
REGISTER_HIP_OPERATOR(
    MaxPool3DGradient,
    PoolGradientOp<float, HIPContext, MaxPool>);
}  // namespace caffe2
