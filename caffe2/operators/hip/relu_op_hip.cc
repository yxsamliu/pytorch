#include "hip/hip_runtime.h"
#include "caffe2/operators/relu_op.h"

#include <algorithm>
#include <functional>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

#ifdef __HIPCC__
typedef __half2 half2;
#endif

template <typename T>
__global__ void ReluHIPKernel(const int N, const T* X, T* Y) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    Y[i] = __ldg(X + i) > 0 ? __ldg(X + i) : T(0);
#else
    Y[i] = X[i] > 0 ? X[i] : T(0);
#endif
  }
}

__global__ void ReluHalfHIPKernel(const int N, const half* X, half* Y) {
  const half kZero = __float2half(0.0f);
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 530
    Y[i] = __hgt(__ldg(X + i), kZero) ? __ldg(X + i) : kZero;
#else
    Y[i] = (__half2float(X[i]) > 0) ? X[i] : kZero;
#endif
  }
}

__global__ void ReluHalf2HIPKernel(const int N, const half2* X, half2* Y) {
  const half2 kZero = __float2half2_rn(0.0f);
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 530
    Y[i] = __hmul2(__hgt2(__ldg(X + i), kZero), __ldg(X + i));
#else
    const float2 xx = __half22float2(X[i]);
    Y[i] = __floats2half2_rn(xx.x > 0 ? xx.x : 0.f, xx.y > 0 ? xx.y : 0.f);
#endif
  }
}

template <typename T>
__global__ void
ReluGradientHIPKernel(const int N, const T* dY, const T* Y, T* dX) {
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 350
    dX[i] = __ldg(Y + i) > 0 ? __ldg(dY + i) : 0;
#else
    dX[i] = Y[i] > 0 ? dY[i] : 0;
#endif
  }
}

__global__ void ReluGradientHalfHIPKernel(
    const int N,
    const half* dY,
    const half* Y,
    half* dX) {
  const half kZero = __float2half(0.0f);
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 530
    dX[i] = __hgt(__ldg(Y + i), kZero) ? __ldg(dY + i) : kZero;
#else
    dX[i] = (__half2float(Y[i]) > 0) ? dY[i] : kZero;
#endif
  }
}

__global__ void ReluGradientHalf2HIPKernel(
    const int N,
    const half2* dY,
    const half2* Y,
    half2* dX) {
  const half2 kZero = __float2half2_rn(0.0f);
  HIP_1D_KERNEL_LOOP(i, N) {
#if __HIP_ARCH__ >= 530
    dX[i] = __hmul2(__hgt2(__ldg(Y + i), kZero), __ldg(dY + i));
#else
    const float2 dy = __half22float2(dY[i]);
    const float2 yy = __half22float2(Y[i]);
    dX[i] = __floats2half2_rn(yy.x > 0 ? dy.x : 0.f, yy.y > 0 ? dy.y : 0.f);
#endif
  }
}

} // namespace

template <>
template <typename T>
bool ReluFunctor<HIPContext>::
operator()(const int N, const T* X, T* Y, HIPContext* context) const {
 hipLaunchKernelGGL( ReluHIPKernel<T>
      , dim3(CAFFE_GET_BLOCKS(N)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), N, X, Y);
  return true;
}

template <>
template <>
bool ReluFunctor<HIPContext>::operator()<at::Half>(
    const int N,
    const at::Half* X,
    at::Half* Y,
    HIPContext* context) const {
  if ((N & 1) == 0) {
   hipLaunchKernelGGL( ReluHalf2HIPKernel, 
        dim3(CAFFE_GET_BLOCKS((N >> 1))),
        dim3(CAFFE_HIP_NUM_THREADS),
        0,
        context->hip_stream(), 
        (N >> 1),
        reinterpret_cast<const half2*>(X),
        reinterpret_cast<half2*>(Y));
  } else {
   hipLaunchKernelGGL( ReluHalfHIPKernel, 
        dim3(CAFFE_GET_BLOCKS(N)),
        dim3(CAFFE_HIP_NUM_THREADS),
        0,
        context->hip_stream(), 
        N, reinterpret_cast<const half*>(X), reinterpret_cast<half*>(Y));
  }
  return true;
}

template <>
template <typename T>
bool ReluGradientFunctor<HIPContext>::Forward(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const T* Y,
    const T* dY,
    T* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
 hipLaunchKernelGGL( ReluGradientHIPKernel<T>
      , dim3(CAFFE_GET_BLOCKS(size)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context->hip_stream(), size, dY, Y, dX);
  return true;
}

template <>
template <>
bool ReluGradientFunctor<HIPContext>::Forward<at::Half>(
    const std::vector<int>& Y_dims,
    const std::vector<int>& /* dY_dims */,
    const at::Half* Y,
    const at::Half* dY,
    at::Half* dX,
    HIPContext* context) const {
  const int size = std::accumulate(
      Y_dims.cbegin(), Y_dims.cend(), 1, std::multiplies<int>());
  if ((size & 1) == 0) {
   hipLaunchKernelGGL( ReluGradientHalf2HIPKernel, 
        dim3(CAFFE_GET_BLOCKS((size >> 1))),
        dim3(CAFFE_HIP_NUM_THREADS),
        0,
        context->hip_stream(), 
        (size >> 1),
        reinterpret_cast<const half2*>(dY),
        reinterpret_cast<const half2*>(Y),
        reinterpret_cast<half2*>(dX));
  } else {
   hipLaunchKernelGGL( ReluGradientHalfHIPKernel, 
        dim3(CAFFE_GET_BLOCKS(size)),
        dim3(CAFFE_HIP_NUM_THREADS),
        0,
        context->hip_stream(), 
        size,
        reinterpret_cast<const half*>(dY),
        reinterpret_cast<const half*>(Y),
        reinterpret_cast<half*>(dX));
  }
  return true;
}

REGISTER_HIP_OPERATOR(
    Relu,
    UnaryElementwiseOp<
        TensorTypes<float, at::Half>,
        HIPContext,
        ReluFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    ReluGradient,
    BinaryElementwiseOp<
        TensorTypes<float, at::Half>,
        HIPContext,
        ReluGradientFunctor<HIPContext>>);

} // namespace caffe2
