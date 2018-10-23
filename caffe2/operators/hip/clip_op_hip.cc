#include "hip/hip_runtime.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/clip_op.h"

namespace caffe2 {
namespace {

template <typename T>
__device__ T hip_min(T x, T y);
template <typename T>
__device__ T hip_max(T x, T y);
template <>
__device__ float hip_min(float x, float y) { return fminf(x, y); }
template <>
__device__ float hip_max(float x, float y) { return fmaxf(x, y); }

// Disabled since we don't use it right now.
/*
template <>
__device__ double hip_min(double x, double y) { return fmin(x, y); }
template <>
__device__ double hip_max(double x, double y) { return fmax(x, y); }
*/


template <typename T>
__global__ void ClipKernel(const int N, const T minval, const T maxval,
                           const T* X, T* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
    Y[i] = hip_min<T>(hip_max<T>(X[i], minval), maxval);
  }
}

template <typename T>
__global__ void ClipGradientKernel(const int N,  const T minval,
                                   const T maxval, const T* Y,
                                   const T* dY, T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
    dX[i] = dY[i] * (Y[i] > minval && Y[i] < maxval);
  }
}
}  // namespace

template <>
bool ClipOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  CAFFE_ENFORCE_GT(X.size(), 0);
  Y->ResizeLike(X);
 hipLaunchKernelGGL( ClipKernel, 
      dim3(CAFFE_GET_BLOCKS(X.size())),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(X.size()), min_, max_, X.data<float>(), Y->template mutable_data<float>());
  return true;
}

template <>
bool ClipGradientOp<float, HIPContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_ENFORCE_GT(Y.size(), 0);
  CAFFE_ENFORCE_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);
 hipLaunchKernelGGL( ClipGradientKernel, 
      dim3(CAFFE_GET_BLOCKS(Y.size())),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(Y.size()),
      min_,
      max_,
      Y.data<float>(),
      dY.data<float>(),
      dX->template mutable_data<float>());
  return true;
}

REGISTER_HIP_OPERATOR(Clip, ClipOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(ClipGradient, ClipGradientOp<float, HIPContext>);
}  // namespace caffe2
