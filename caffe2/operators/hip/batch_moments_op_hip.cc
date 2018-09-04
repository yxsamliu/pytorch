#include "hip/hip_runtime.h"
#include "caffe2/operators/batch_moments_op.h"

#include <cub/block/block_reduce.cuh>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_HIP_NUM_THREADS>;

template <typename T, StorageOrder kOrder>
__global__ void BatchMomentsHIPKernel(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T* mu,
    T* var) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  __shared__ typename BlockReduce<T>::TempStorage m_storage;
  __shared__ typename BlockReduce<T>::TempStorage v_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T m_sum = 0;
    T v_sum = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = kOrder == StorageOrder::NCHW
          ? (j / HxW * C + i) * HxW + j % HxW
          : j * outer_size + i;
#if __HIP_ARCH__ >= 350
      m_sum += __ldg(X + index);
      v_sum += __ldg(X + index) * __ldg(X + index);
#else
      m_sum += X[index];
      v_sum += X[index] * X[index];
#endif
    }
    m_sum = BlockReduce<T>(m_storage).Reduce(m_sum, cub::Sum());
    v_sum = BlockReduce<T>(v_storage).Reduce(v_sum, cub::Sum());
    if (threadIdx.x == 0) {
      mu[i] = m_sum / static_cast<T>(N * HxW);
      var[i] = v_sum / static_cast<T>(N * HxW);
    }
    __syncthreads();
  }
}

template <typename T, StorageOrder kOrder>
__global__ void BatchMomentsGradientHIPKernel(
    const int N,
    const int C,
    const int HxW,
    const T* dmu,
    const T* dvar,
    const T* X,
    T* dX) {
  const int size = N * C * HxW;
  const T scale = T(1) / static_cast<T>(N * HxW);
  HIP_1D_KERNEL_LOOP(i, size) {
    const int i_mu = kOrder == StorageOrder::NCHW ? i / (HxW) % C : i % C;
#if __HIP_ARCH__ >= 350
    dX[i] =
        (__ldg(dmu + i_mu) + __ldg(dvar + i_mu) * T(2) * __ldg(X + i)) * scale;
#else
    dX[i] = (dmu[i_mu] + dvar[i_mu] * T(2) * X[i]) * scale;
#endif
  }
}

} // namespace

template <>
bool BatchMomentsOp<float, HIPContext>::ComputeBatchMomentsNCHW(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    float* mu,
    float* var) {
  const int outer_size = N * HxW;
 hipLaunchKernelGGL( BatchMomentsHIPKernel<float, StorageOrder::NCHW>
      , dim3(std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context_.hip_stream(), N, C, HxW, X, mu, var);
  return true;
}

template <>
bool BatchMomentsOp<float, HIPContext>::ComputeBatchMomentsNHWC(
    const int N,
    const int C,
    const int HxW,
    const float* X,
    float* mu,
    float* var) {
  const int outer_size = N * HxW;
 hipLaunchKernelGGL( BatchMomentsHIPKernel<float, StorageOrder::NHWC>
      , dim3(std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context_.hip_stream(), N, C, HxW, X, mu, var);
  return true;
}

template <>
bool BatchMomentsGradientOp<float, HIPContext>::
    ComputeBatchMomentsGradientNCHW(
        const int N,
        const int C,
        const int HxW,
        const float* dmu,
        const float* dvar,
        const float* X,
        float* dX) {
  const int size = N * C * HxW;
 hipLaunchKernelGGL( BatchMomentsGradientHIPKernel<float, StorageOrder::NCHW>
      , dim3(CAFFE_GET_BLOCKS(size)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context_.hip_stream(), N, C, HxW, dmu, dvar, X, dX);
  return true;
}

template <>
bool BatchMomentsGradientOp<float, HIPContext>::
    ComputeBatchMomentsGradientNHWC(
        const int N,
        const int C,
        const int HxW,
        const float* dmu,
        const float* dvar,
        const float* X,
        float* dX) {
  const int size = N * C * HxW;
 hipLaunchKernelGGL( BatchMomentsGradientHIPKernel<float, StorageOrder::NHWC>
      , dim3(CAFFE_GET_BLOCKS(size)),
         dim3(CAFFE_HIP_NUM_THREADS),
         0,
         context_.hip_stream(), N, C, HxW, dmu, dvar, X, dX);
  return true;
}

REGISTER_HIP_OPERATOR(BatchMoments, BatchMomentsOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    BatchMomentsGradient,
    BatchMomentsGradientOp<float, HIPContext>);

} // namespace caffe2
