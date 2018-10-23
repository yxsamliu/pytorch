#include "hip/hip_runtime.h"
#include <cub/block/block_reduce.cuh>
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/reduce_front_back_max_ops.h"

namespace caffe2 {

/***
  Max Ops
***/

namespace {

__global__ void columnwise_max_kernel(
    const int rows,
    const int cols,
    const float* data,
    const int* lengths,
    float* out) {
  typedef cub::BlockReduce<float, CAFFE_HIP_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int colIndex = blockIdx.x; colIndex < cols; colIndex += gridDim.x) {
    float mx = FLT_MIN;
    const int length = lengths == nullptr ? rows : lengths[colIndex];
    for (int rowIndex = threadIdx.x; rowIndex < length;
         rowIndex += blockDim.x) {
      mx = fmaxf(mx, data[rowIndex * cols + colIndex]);
    }
    mx = BlockReduce(temp_storage).Reduce(mx, cub::Max());
    if (threadIdx.x == 0) {
      out[colIndex] = mx;
    }
    __syncthreads();
  }
}

__global__ void rowwise_max_kernel(
    const int rows,
    const int cols,
    const float* data,
    const int* lengths,
    float* out) {
  typedef cub::BlockReduce<float, CAFFE_HIP_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int rowIndex = blockIdx.x; rowIndex < rows; rowIndex += gridDim.x) {
    float mx = FLT_MIN;
    const int length = lengths == nullptr ? cols : lengths[rowIndex];
    for (int colIndex = threadIdx.x; colIndex < length;
         colIndex += blockDim.x) {
      mx = fmaxf(mx, data[rowIndex * cols + colIndex]);
    }
    mx = BlockReduce(temp_storage).Reduce(mx, cub::Max());
    if (threadIdx.x == 0) {
      out[rowIndex] = mx;
    }
    __syncthreads();
  }
}

__global__ void columnwise_max_grad_kernel(
    const int rows,
    const int cols,
    const float* dYdata,
    const float* Xdata,
    const float* Ydata,
    const int* lengths,
    float* dXdata) {
  HIP_1D_KERNEL_LOOP(i, rows * cols) {
    int col = i % cols;
    int row = i / cols;
    if (lengths != nullptr && row >= lengths[col]) {
      dXdata[i] = 0.0f;
    } else {
      dXdata[i] = (Xdata[i] == Ydata[col]) * dYdata[col];
    }
  }
}

__global__ void rowwise_max_grad_kernel(
    const int rows,
    const int cols,
    const float* dYdata,
    const float* Xdata,
    const float* Ydata,
    const int* lengths,
    float* dXdata) {
  HIP_1D_KERNEL_LOOP(i, rows * cols) {
    int col = i % cols;
    int row = i / cols;
    if (lengths != nullptr && col >= lengths[row]) {
      dXdata[i] = 0.0f;
    } else {
      dXdata[i] = (Xdata[i] == Ydata[row]) * dYdata[row];
    }
  }
}
} // anonymous namespace

// ReduceFrontmax
template <>
void MaxReduceDimsOp<float, HIPContext, true>::Compute(
    int rows,
    int cols,
    const float* data,
    const int32_t* lengths_data,
    float* out_data) {
 hipLaunchKernelGGL( columnwise_max_kernel, 
      dim3(::min(cols, CAFFE_MAXIMUM_NUM_BLOCKS)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), static_cast<const int>(rows), static_cast<const int>(cols), data, lengths_data, out_data);
}

// ReduceBackMax
template <>
void MaxReduceDimsOp<float, HIPContext, false>::Compute(
    int rows,
    int cols,
    const float* data,
    const int32_t* lengths_data,
    float* out_data) {
 hipLaunchKernelGGL( rowwise_max_kernel, 
      dim3(::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), static_cast<const int>(rows), static_cast<const int>(cols), data, lengths_data, out_data);
}

// ReduceFrontMaxGradient
template <>
void MaxReduceDimsGradientOp<float, HIPContext, true>::Compute(
    int rows,
    int cols,
    const float* dYdata,
    const float* Xdata,
    const float* Ydata,
    const int32_t* lengths_data,
    float* dXdata) {
 hipLaunchKernelGGL( columnwise_max_grad_kernel, 
      dim3(CAFFE_GET_BLOCKS(rows * cols)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(rows), static_cast<const int>(cols), dYdata, Xdata, Ydata, lengths_data, dXdata);
}

// ReduceBackMaxGradient
template <>
void MaxReduceDimsGradientOp<float, HIPContext, false>::Compute(
    int rows,
    int cols,
    const float* dYdata,
    const float* Xdata,
    const float* Ydata,
    const int* lengths_data,
    float* dXdata) {
 hipLaunchKernelGGL( rowwise_max_grad_kernel, 
      dim3(CAFFE_GET_BLOCKS(rows * cols)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(rows), static_cast<const int>(cols), dYdata, Xdata, Ydata, lengths_data, dXdata);
}

REGISTER_HIP_OPERATOR(
    ReduceFrontMax,
    MaxReduceDimsOp<float, HIPContext, true>);
REGISTER_HIP_OPERATOR(
    ReduceFrontMaxGradient,
    MaxReduceDimsGradientOp<float, HIPContext, true>);

REGISTER_HIP_OPERATOR(
    ReduceBackMax,
    MaxReduceDimsOp<float, HIPContext, false>);
REGISTER_HIP_OPERATOR(
    ReduceBackMaxGradient,
    MaxReduceDimsGradientOp<float, HIPContext, false>);

} // namespace caffe2
