#include "hip/hip_runtime.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/lstm_unit_op.h"

namespace caffe2 {

namespace detail {

template <typename Dtype>
__device__ Dtype hip_sigmoid(const Dtype x) {
  return Dtype(1) / (Dtype(1) + exp(-x));
}

template <typename T, typename MATH>
__global__ void LSTMUnitKernel(
    const int nthreads,
    const int dim,
    const int t,
    const T* H_prev,
    const T* C_prev,
    const T* X,
    const int32_t* seqLengths,
    bool drop_states,
    T* C,
    T* H,
    const MATH forget_bias) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const int d = index % dim;
    const bool valid = seqLengths == nullptr || t < seqLengths[n];
    if (!valid) {
      H[index] = convert::To<MATH, T>(convert::To<T, MATH>(H_prev[index]) * !drop_states);
      C[index] = convert::To<MATH, T>(convert::To<T, MATH>(C_prev[index]) * !drop_states);
    } else {
      const T* X_offset = X + 4 * dim * n;
      const MATH i = hip_sigmoid(convert::To<T, MATH>(X_offset[d]));
      const MATH f = hip_sigmoid(convert::To<T, MATH>(X_offset[1 * dim + d]) + forget_bias);
      const MATH o = hip_sigmoid(convert::To<T, MATH>(X_offset[2 * dim + d]));
      const MATH g = tanh(convert::To<T, MATH>(X_offset[3 * dim + d]));
      const MATH c_prev = convert::To<T, MATH>(C_prev[index]);
      const MATH c = f * c_prev + i * g;
      C[index] = convert::To<MATH, T>(c);
      const MATH tanh_c = tanh(c);
      H[index] = convert::To<MATH, T>(o * tanh_c);
    }
  }
}

template <typename T, typename MATH>
__global__ void LSTMUnitGradientKernel(
    const int nthreads,
    const int dim,
    const int t,
    const T* C_prev,
    const T* X,
    const T* C,
    const T* H,
    const int32_t* seqLengths,
    const T* C_diff,
    const T* H_diff,
    bool drop_states,
    T* H_prev_diff,
    T* C_prev_diff,
    T* X_diff,
    const MATH forget_bias) {
  HIP_1D_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const bool valid = seqLengths == nullptr || t < seqLengths[n];
    const int d = index % dim;
    const T* X_offset = X + 4 * dim * n;
    T* c_prev_diff = C_prev_diff + index;
    T* h_prev_diff = H_prev_diff + index;
    T* X_diff_offset = X_diff + 4 * dim * n;
    T* i_diff = X_diff_offset + d;
    T* f_diff = X_diff_offset + 1 * dim + d;
    T* o_diff = X_diff_offset + 2 * dim + d;
    T* g_diff = X_diff_offset + 3 * dim + d;
    if (!valid) {
      *h_prev_diff = convert::To<MATH, T>(convert::To<T, MATH>(H_diff[index]) *
                                          !drop_states);
      *c_prev_diff = convert::To<MATH, T>(convert::To<T, MATH>(C_diff[index]) *
                                          !drop_states);
      *i_diff = convert::To<MATH, T>(0);
      *f_diff = convert::To<MATH, T>(0);
      *o_diff = convert::To<MATH, T>(0);
      *g_diff = convert::To<MATH, T>(0);
    } else {
      const MATH i = hip_sigmoid(convert::To<T, MATH>(X_offset[d]));
      const MATH f = hip_sigmoid(convert::To<T, MATH>(X_offset[1 * dim + d]) + forget_bias);
      const MATH o = hip_sigmoid(convert::To<T, MATH>(X_offset[2 * dim + d]));
      const MATH g = tanh(convert::To<T, MATH>(X_offset[3 * dim + d]));
      const MATH c_prev = convert::To<T, MATH>(C_prev[index]);
      const MATH c = convert::To<T, MATH>(C[index]);
      const MATH tanh_c = tanh(c);
      const MATH c_term_diff =
          convert::To<T, MATH>(C_diff[index]) +
          convert::To<T, MATH>(H_diff[index]) * o * (1 - tanh_c * tanh_c);
      *c_prev_diff = convert::To<MATH, T>(c_term_diff * f);
      *h_prev_diff = convert::To<MATH, T>(0);
      *i_diff = convert::To<MATH, T>(c_term_diff * g * i * (1 - i));
      *f_diff = convert::To<MATH, T>(c_term_diff * c_prev * f * (1 - f));
      *o_diff = convert::To<MATH, T>(
                  convert::To<T, MATH>(H_diff[index]) * tanh_c * o * (1 - o));
      *g_diff = convert::To<MATH, T>(c_term_diff * i * (1 - g * g));
    }
  }
}

template <>
void LSTMUnit<float, HIPContext>(
    int N,
    int D,
    int t,
    const float* H_prev,
    const float* C_prev,
    const float* X,
    const int32_t* seqLengths,
    bool drop_states,
    float* C,
    float* H,
    const float forget_bias,
    HIPContext* context) {
 hipLaunchKernelGGL( LSTMUnitKernel<float, float>, 
      dim3(CAFFE_GET_BLOCKS(N * D)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(), 
      static_cast<const int>(N * static_cast<const int>(D)),
      static_cast<const int>(D),
      static_cast<const int>(t),
      H_prev,
      C_prev,
      X,
      seqLengths,
      drop_states,
      C,
      H,
      forget_bias);
}

template <>
void LSTMUnit<float16, HIPContext>(
    int N,
    int D,
    int t,
    const float16* H_prev,
    const float16* C_prev,
    const float16* X,
    const int32_t* seqLengths,
    bool drop_states,
    float16* C,
    float16* H,
    const float forget_bias,
    HIPContext* context) {
 hipLaunchKernelGGL( LSTMUnitKernel<float16, float>, 
      dim3(CAFFE_GET_BLOCKS(N * D)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(), 
      static_cast<const int>(N * static_cast<const int>(D)),
      static_cast<const int>(D),
      static_cast<const int>(t),
      H_prev,
      C_prev,
      X,
      seqLengths,
      drop_states,
      C,
      H,
      forget_bias);
}

template <>
void LSTMUnitGradient<float, HIPContext>(
    int N,
    int D,
    int t,
    const float* C_prev,
    const float* X,
    const int32_t* seqLengths,
    const float* C,
    const float* H,
    const float* C_diff,
    const float* H_diff,
    bool drop_states,
    float* H_prev_diff,
    float* C_prev_diff,
    float* X_diff,
    const float forget_bias,
    HIPContext* context) {
 hipLaunchKernelGGL( LSTMUnitGradientKernel<float, float>, 
      dim3(CAFFE_GET_BLOCKS(N * D)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(), 
      static_cast<const int>(N * static_cast<const int>(D)),
      static_cast<const int>(D),
      static_cast<const int>(t),
      C_prev,
      X,
      C,
      H,
      seqLengths,
      C_diff,
      H_diff,
      drop_states,
      H_prev_diff,
      C_prev_diff,
      X_diff,
      forget_bias);
}

template <>
void LSTMUnitGradient<float16, HIPContext>(
    int N,
    int D,
    int t,
    const float16* C_prev,
    const float16* X,
    const int32_t* seqLengths,
    const float16* C,
    const float16* H,
    const float16* C_diff,
    const float16* H_diff,
    bool drop_states,
    float16* H_prev_diff,
    float16* C_prev_diff,
    float16* X_diff,
    const float forget_bias,
    HIPContext* context) {
 hipLaunchKernelGGL( LSTMUnitGradientKernel<float16, float>, 
      dim3(CAFFE_GET_BLOCKS(N * D)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(), 
      static_cast<const int>(N * static_cast<const int>(D)),
      static_cast<const int>(D),
      static_cast<const int>(t),
      C_prev,
      X,
      C,
      H,
      seqLengths,
      C_diff,
      H_diff,
      drop_states,
      H_prev_diff,
      C_prev_diff,
      X_diff,
      forget_bias);
}
}

template <>
bool LSTMUnitOp<HIPContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(0));
}

template <>
bool LSTMUnitGradientOp<HIPContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<float, float16>>::call(this, Input(0));
}

REGISTER_HIP_OPERATOR(LSTMUnit, LSTMUnitOp<HIPContext>);
REGISTER_HIP_OPERATOR(
    LSTMUnitGradient,
    LSTMUnitGradientOp<HIPContext>);
}
