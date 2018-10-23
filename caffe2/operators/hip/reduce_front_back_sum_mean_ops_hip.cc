#include "hip/hip_runtime.h"
#include <cub/block/block_reduce.cuh>
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/reduce_front_back_sum_mean_ops.h"

namespace caffe2 {

namespace {
template <typename T, bool NORMALIZE>
__global__ void columnwise_fill_kernel(
    const int rows,
    const int cols,
    const T* dY,
    const int* lengths,
    T* dX) {
  HIP_1D_KERNEL_LOOP(i, rows * cols) {
    int row = i / cols;
    int col = i % cols;
    if (lengths == nullptr) {
      dX[i] = NORMALIZE ? dY[col] / rows : dY[col];
    } else if (row < lengths[col]) {
      dX[i] = NORMALIZE ? dY[col] / lengths[col] : dY[col];
    } else {
      dX[i] = 0;
    }
  }
}

template <typename T, bool NORMALIZE>
__global__ void rowwise_fill_kernel(
    const int rows,
    const int cols,
    const T* dY,
    const int* lengths,
    T* dX) {
  HIP_1D_KERNEL_LOOP(i, rows * cols) {
    int row = i / cols;
    int col = i % cols;
    if (lengths == nullptr) {
      dX[i] = NORMALIZE ? dY[row] / cols : dY[row];
    } else if (col < lengths[row]) {
      dX[i] = NORMALIZE ? dY[row] / lengths[row] : dY[row];
    } else {
      dX[i] = 0;
    }
  }
}

template <typename T, bool NORMALIZE>
__global__ void rowwise_sum_kernel(
    const int rows,
    const int cols,
    const T* data,
    const int* lengths,
    T* out) {
  typedef cub::BlockReduce<float, CAFFE_HIP_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int rowIndex = blockIdx.x; rowIndex < rows; rowIndex += gridDim.x) {
    T sum = 0;
    const int rowOffset = rowIndex * cols;
    const int length = lengths == nullptr ? cols : lengths[rowIndex];
    for (int colIndex = threadIdx.x; colIndex < length;
         colIndex += blockDim.x) {
      sum += data[rowOffset + colIndex];
    }
    sum = BlockReduce(temp_storage).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
      out[rowIndex] = NORMALIZE ? sum / length : sum;
    }
    __syncthreads();
  }
}

template <typename T, bool NORMALIZE>
__global__ void columnwise_sum_kernel(
    const int rows,
    const int cols,
    const T* data,
    const int* lengths,
    T* out) {
  typedef cub::BlockReduce<float, CAFFE_HIP_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  for (int colIndex = blockIdx.x; colIndex < cols; colIndex += gridDim.x) {
    T sum = 0;
    const int length = lengths == nullptr ? rows : lengths[colIndex];
    for (int rowIndex = threadIdx.x; rowIndex < length;
         rowIndex += blockDim.x) {
      sum += data[rowIndex * cols + colIndex];
    }
    sum = BlockReduce(temp_storage).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
      out[colIndex] = NORMALIZE ? sum / length : sum;
    }
    __syncthreads();
  }
}

} // anonymous namespace

/***
  Sum Ops
***/

// ReduceFrontSum: columnwise sum
template <>
template <typename T>
void SumReduceDimsOp<HIPContext, true, false>::Compute(
    int rows,
    int cols,
    const T* in_data,
    const int* lengths_data,
    T* out_data) {
 hipLaunchKernelGGL( columnwise_sum_kernel<T, false>
      , dim3(::min(cols, CAFFE_MAXIMUM_NUM_BLOCKS)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context_.hip_stream(), static_cast<const int>(rows), static_cast<const int>(cols), in_data, lengths_data, out_data);
}

// ReduceBackSum: rowwise sum
template <>
template <typename T>
void SumReduceDimsOp<HIPContext, false, false>::Compute(
    int rows,
    int cols,
    const T* in_data,
    const int* lengths_data,
    T* out_data) {
 hipLaunchKernelGGL( rowwise_sum_kernel<T, false>
      , dim3(::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context_.hip_stream(), static_cast<const int>(rows), static_cast<const int>(cols), in_data, lengths_data, out_data);
}

// ReduceFrontSumGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<HIPContext, true, false>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    const int* lengths_data,
    T* dXdata) {
 hipLaunchKernelGGL( columnwise_fill_kernel<T, false>
      , dim3(CAFFE_GET_BLOCKS(rows * cols)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context_.hip_stream(), static_cast<const int>(rows), static_cast<const int>(cols), dYdata, lengths_data, dXdata);
}

// ReduceBackSumGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<HIPContext, false, false>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    const int* lengths_data,
    T* dXdata) {
 hipLaunchKernelGGL( rowwise_fill_kernel<T, false>
      , dim3(CAFFE_GET_BLOCKS(rows * cols)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context_.hip_stream(), static_cast<const int>(rows), static_cast<const int>(cols), dYdata, lengths_data, dXdata);
}

REGISTER_HIP_OPERATOR(
    ReduceFrontSum,
    SumReduceDimsOp<HIPContext, true, false>);
REGISTER_HIP_OPERATOR(
    ReduceFrontSumGradient,
    SumReduceDimsGradientOp<HIPContext, true, false>);

REGISTER_HIP_OPERATOR(
    ReduceBackSum,
    SumReduceDimsOp<HIPContext, false, false>);
REGISTER_HIP_OPERATOR(
    ReduceBackSumGradient,
    SumReduceDimsGradientOp<HIPContext, false, false>);

/***
  Mean Ops
***/

// ReduceFrontMean: columnwise mean
template <>
template <typename T>
void SumReduceDimsOp<HIPContext, true, true>::Compute(
    int rows,
    int cols,
    const T* in_data,
    const int* lengths_data,
    T* out_data) {
 hipLaunchKernelGGL( columnwise_sum_kernel<T, true>
      , dim3(::min(cols, CAFFE_MAXIMUM_NUM_BLOCKS)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context_.hip_stream(), static_cast<const int>(rows), static_cast<const int>(cols), in_data, lengths_data, out_data);
}

// ReduceBackMean: rowwise mean
template <>
template <typename T>
void SumReduceDimsOp<HIPContext, false, true>::Compute(
    int rows,
    int cols,
    const T* in_data,
    const int* lengths_data,
    T* out_data) {
 hipLaunchKernelGGL( rowwise_sum_kernel<T, true>
      , dim3(::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context_.hip_stream(), static_cast<const int>(rows), static_cast<const int>(cols), in_data, lengths_data, out_data);
}

// ReduceFrontMeanGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<HIPContext, true, true>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    const int* lengths_data,
    T* dXdata) {
 hipLaunchKernelGGL( columnwise_fill_kernel<T, true>
      , dim3(CAFFE_GET_BLOCKS(rows * cols)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context_.hip_stream(), static_cast<const int>(rows), static_cast<const int>(cols), dYdata, lengths_data, dXdata);
}

// ReduceBackMeanGradient
template <>
template <typename T>
void SumReduceDimsGradientOp<HIPContext, false, true>::Compute(
    int rows,
    int cols,
    const T* dYdata,
    const int* lengths_data,
    T* dXdata) {
 hipLaunchKernelGGL( rowwise_fill_kernel<T, true>
      , dim3(CAFFE_GET_BLOCKS(rows * cols)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context_.hip_stream(), static_cast<const int>(rows), static_cast<const int>(cols), dYdata, lengths_data, dXdata);
}

REGISTER_HIP_OPERATOR(
    ReduceFrontMean,
    SumReduceDimsOp<HIPContext, true, true>);
REGISTER_HIP_OPERATOR(
    ReduceFrontMeanGradient,
    SumReduceDimsGradientOp<HIPContext, true, true>);

REGISTER_HIP_OPERATOR(
    ReduceBackMean,
    SumReduceDimsOp<HIPContext, false, true>);
REGISTER_HIP_OPERATOR(
    ReduceBackMeanGradient,
    SumReduceDimsGradientOp<HIPContext, false, true>);

} // namespace caffe2
