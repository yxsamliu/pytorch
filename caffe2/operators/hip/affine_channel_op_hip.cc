#include "hip/hip_runtime.h"
#include "caffe2/operators/affine_channel_op.h"

#include <cub/block/block_reduce.cuh>

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

namespace {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_HIP_NUM_THREADS>;

template <typename T, StorageOrder kOrder>
__global__ void AffineChannelScaleBiasBackwardHIPKernel(
    const int N,
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    T* dscale,
    T* dbias) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  __shared__ typename BlockReduce<T>::TempStorage ds_storage;
  __shared__ typename BlockReduce<T>::TempStorage db_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T ds_sum = 0;
    T db_sum = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = kOrder == StorageOrder::NCHW
          ? (j / HxW * C + i) * HxW + j % HxW
          : j * outer_size + i;
#if __HIP_ARCH__ >= 350
      ds_sum += __ldg(dY + index) * __ldg(X + index);
      db_sum += __ldg(dY + index);
#else
      ds_sum += dY[index] * X[index];
      db_sum += dY[index];
#endif
    }
    ds_sum = BlockReduce<T>(ds_storage).Reduce(ds_sum, cub::Sum());
    db_sum = BlockReduce<T>(db_storage).Reduce(db_sum, cub::Sum());
    if (threadIdx.x == 0) {
      dscale[i] = ds_sum;
      dbias[i] = db_sum;
    }
    __syncthreads();
  }
}

} // namespace

template <>
bool AffineChannelGradientOp<float, HIPContext>::RunOnDeviceWithOrderNCHW() {
  const auto& dY = Input(0);
  const auto& scale = is_learnable_ ? Input(2) : Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(dY);
  const int N = dY.dim32(0);
  const int C = dY.dim32(1);
  const int HxW = dY.size() / (N * C);
  const float* dY_data = dY.data<float>();
  const float* scale_data = scale.data<float>();
  const std::array<int, 3> X_dims = {N, C, HxW};
  const std::array<int, 3> scale_dims = {1, C, 1};
  math::Mul<float, HIPContext>(
      3,
      X_dims.data(),
      3,
      scale_dims.data(),
      dY_data,
      scale_data,
      dX->template mutable_data<float>(),
      &context_);
  if (is_learnable_) {
    const auto& X = Input(1);
    const float* X_data = X.data<float>();
    auto* dscale = Output(1);
    auto* dbias = Output(2);
    dscale->ResizeLike(scale);
    dbias->ResizeLike(scale);
    const int outer_size = N * HxW;
   hipLaunchKernelGGL( AffineChannelScaleBiasBackwardHIPKernel<float, StorageOrder::NCHW>
        , dim3(::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS)),
           dim3(CAFFE_HIP_NUM_THREADS),
           0,
           context_.hip_stream(), 
            N,
            C,
            HxW,
            dY_data,
            X_data,
            dscale->template mutable_data<float>(),
            dbias->template mutable_data<float>());
  }
  return true;
}

template <>
bool AffineChannelGradientOp<float, HIPContext>::RunOnDeviceWithOrderNHWC() {
  const auto& dY = Input(0);
  const auto& scale = is_learnable_ ? Input(2) : Input(1);
  auto* dX = Output(0);
  dX->ResizeLike(dY);
  const int ndim = dY.ndim();
  const int C = dY.dim32(ndim - 1);
  const int rows = dY.size() / C;
  const int cols = C;
  const float* dY_data = dY.data<float>();
  const float* scale_data = scale.data<float>();
  math::RowwiseMul<float, HIPContext>(
      rows,
      cols,
      dY_data,
      scale_data,
      dX->template mutable_data<float>(),
      &context_);
  if (is_learnable_) {
    const auto& X = Input(1);
    const float* X_data = X.data<float>();
    const int N = X.dim32(0);
    const int HxW = rows / N;
    auto* dscale = Output(1);
    auto* dbias = Output(2);
    dscale->ResizeLike(scale);
    dbias->ResizeLike(scale);
   hipLaunchKernelGGL( AffineChannelScaleBiasBackwardHIPKernel<float, StorageOrder::NHWC>
        , dim3(::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS)),
           dim3(CAFFE_HIP_NUM_THREADS),
           0,
           context_.hip_stream(), 
            N,
            C,
            HxW,
            dY_data,
            X_data,
            dscale->template mutable_data<float>(),
            dbias->template mutable_data<float>());
  }
  return true;
}

REGISTER_HIP_OPERATOR(AffineChannel, AffineChannelOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    AffineChannelGradient,
    AffineChannelGradientOp<float, HIPContext>);

} // namespace caffe2
