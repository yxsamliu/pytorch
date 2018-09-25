#include "hip/hip_runtime.h"
#include <cub/block/block_reduce.cuh>

#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/normalize_l1_op.h"
#include "caffe2/operators/normalize_op.h"

namespace caffe2 {

__global__ void NormalizeKernel(
    const int m,
    const int n,
    const int sf,
    const float* xData,
    float* yData,
    const float kEps) {
  typedef cub::BlockReduce<float, CAFFE_HIP_NUM_THREADS> BlockReduce;
  __shared__ BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.x; i < n; i += gridDim.x) {
    auto base = (i / sf) * sf * m + (i % sf);

    float sum = 0.0;
    __shared__ float norm;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
      const auto x_ij = xData[base + j * sf];
      sum += x_ij * x_ij;
    }
    float reduce_result = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
      norm = sqrtf(reduce_result);
      norm = fmaxf(norm, kEps);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
      const auto index = base + j * sf;
      yData[index] = xData[index] / norm;
    }
  }
}

__global__ void NormalizeGradientKernel(
    const int M,
    const int N,
    const int SF,
    const float* in_mat,
    const float* grad_out_mat,
    float* grad_mat,
    const float kEps) {
  typedef cub::BlockReduce<float, CAFFE_HIP_NUM_THREADS> BlockReduce;
  __shared__ BlockReduce::TempStorage temp_storage_sum;
  __shared__ BlockReduce::TempStorage temp_storage_norm;
  for (int i = blockIdx.x; i < M; i += gridDim.x) {
    float sum = 0.0;
    float norm = 0.0;
    __shared__ float row_sum;
    __shared__ float row_norm;
    __shared__ float row_norm_3;
    auto base = (i / SF) * SF * N + (i % SF);
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      int index = base + j * SF;
      sum += in_mat[index] * grad_out_mat[index];
      norm += in_mat[index] * in_mat[index];
    }
    float reduce_result = BlockReduce(temp_storage_sum).Sum(sum);
    float reduce_norm = BlockReduce(temp_storage_norm).Sum(norm);

    if (threadIdx.x == 0) {
      row_sum = reduce_result;
      row_norm = sqrtf(reduce_norm);
      row_norm = fmaxf(row_norm, kEps);
      row_norm_3 = powf(row_norm, 3);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
      int index = base + j * SF;
      const float x_ij = in_mat[index];
      const float dy_ij = grad_out_mat[index];
      grad_mat[index] = (dy_ij / row_norm) - ((x_ij / row_norm_3) * row_sum);
    }
  }
}

template <>
void NormalizeOp<float, HIPContext>::DoNormalize(
    const float* xData,
    float* yData,
    const int m,
    const int n,
    const int sf) {
 hipLaunchKernelGGL( NormalizeKernel, 
      dim3(min(n, CAFFE_MAXIMUM_NUM_BLOCKS)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), static_cast<const int>(m), static_cast<const int>(n), static_cast<const int>(sf), xData, yData, kEps_);
}

template <>
bool NormalizeGradientOp<float, HIPContext>::RunOnDevice() {
  const auto& X = Input(0);
  const auto& dY = Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(X);

  const auto canonical_axis =
      X.canonical_axis_index(OperatorBase::GetSingleArgument<int>("axis", -1));
  int N = X.dim32(canonical_axis);
  int M = X.size() / N;
  const int SF = X.size_from_dim(canonical_axis + 1);
 hipLaunchKernelGGL( NormalizeGradientKernel, 
      dim3(min(M, CAFFE_MAXIMUM_NUM_BLOCKS)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(M),
      static_cast<const int>(N),
      static_cast<const int>(SF),
      X.data<float>(),
      dY.data<float>(),
      dX->template mutable_data<float>(),
      kEps_);
  return true;
}

namespace {
__global__ void NormalizeL1Kernel(
    const int m,
    const int n,
    const int sf,
    const float* xData,
    float* yData) {
  typedef cub::BlockReduce<float, CAFFE_HIP_NUM_THREADS> BlockReduce;
  __shared__ BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.x; i < n; i += gridDim.x) {
    auto base = (i / sf) * sf * m + (i % sf);

    float sum = 0.0;
    __shared__ float norm;
    for (int j = threadIdx.x; j < m; j += blockDim.x) {
      const auto x_ij = xData[base + j * sf];
      sum += fabsf(x_ij);
    }
    float reduce_result = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
      norm = reduce_result;
    }
    __syncthreads();
    if (norm != 0) {
      for (int j = threadIdx.x; j < m; j += blockDim.x) {
        const auto index = base + j * sf;
        yData[index] = xData[index] / norm;
      }
    }
  }
}
} // namespace

template <>
void NormalizeL1Op<float, HIPContext>::DoNormalize(
    const float* xData,
    float* yData,
    const int m,
    const int n,
    const int sf) {
 hipLaunchKernelGGL( NormalizeL1Kernel, 
      dim3(min(n, CAFFE_MAXIMUM_NUM_BLOCKS)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), static_cast<const int>(m), static_cast<const int>(n), static_cast<const int>(sf), xData, yData);
}

REGISTER_HIP_OPERATOR(Normalize, NormalizeOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    NormalizeGradient,
    NormalizeGradientOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(NormalizeL1, NormalizeL1Op<float, HIPContext>);
} // namespace caffe2
