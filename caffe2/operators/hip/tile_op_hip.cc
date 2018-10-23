#include "hip/hip_runtime.h"
#include <cub/block/block_reduce.cuh>

#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/tile_op.h"

namespace caffe2 {
namespace {
__global__ void TileCopyKernel(
    int item_size,
    int outer_dim,
    int inner_dim,
    int tiles,
    const char* input_data,
    char* output_data) {
  HIP_1D_KERNEL_LOOP(index, outer_dim * tiles) {
    int i = index / tiles;
    int t = index % tiles;
    const char* input_ptr = input_data + inner_dim * item_size * i;
    char* output_ptr = output_data + (i * tiles + t) * inner_dim * item_size;
    memcpy(output_ptr, input_ptr, inner_dim * item_size);
  }
}

template <typename T>
__global__ void TileGradientAxpyKernel(
    int outer_dim,
    int inner_dim,
    int tiles,
    const T* input_data,
    T* output_data) {
  typedef cub::BlockReduce<T, CAFFE_HIP_NUM_THREADS> BlockReduce;

  for (int idx = blockIdx.x; idx < outer_dim * inner_dim; idx += gridDim.x) {
    int i = idx / inner_dim;
    int j = idx % inner_dim;
    T* output_ptr = output_data + inner_dim * i;

    T x = 0.0;
    for (int t = threadIdx.x; t < tiles; t += blockDim.x) {
      const T* input_ptr = input_data + (i * tiles + t) * inner_dim;
      x += input_ptr[j];
    }
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T totx = BlockReduce(temp_storage).Sum(x);
    if (threadIdx.x == 0) {
      output_ptr[j] = totx;
    }
    __syncthreads();
  }
}
} // namespace

template <>
void TileOp<HIPContext>::DoTile(
    const TypeMeta& meta,
    int item_size,
    int outer_dim,
    int inner_dim,
    const char* input_data,
    char* output_data) {
 hipLaunchKernelGGL( TileCopyKernel, 
      dim3(::min(outer_dim * tiles_, CAFFE_MAXIMUM_NUM_BLOCKS)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<int>(item_size), static_cast<int>(outer_dim), static_cast<int>(inner_dim), static_cast<int>(tiles_), input_data, output_data);
}

template <>
void TileGradientOp<float, HIPContext>::DoTileGradient(
    const TypeMeta& meta,
    int item_size,
    int outer_dim,
    int inner_dim,
    const char* input_data,
    char* output_data) {
 hipLaunchKernelGGL( TileGradientAxpyKernel<float>, 
      dim3(::min(outer_dim * inner_dim, CAFFE_MAXIMUM_NUM_BLOCKS)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<int>(outer_dim),
      static_cast<int>(inner_dim),
      static_cast<int>(tiles_),
      reinterpret_cast<const float*>(input_data),
      reinterpret_cast<float*>(output_data));
}

REGISTER_HIP_OPERATOR(Tile, TileOp<HIPContext>);
REGISTER_HIP_OPERATOR(TileGradient, TileGradientOp<float, HIPContext>);
} // namespace caffe2
