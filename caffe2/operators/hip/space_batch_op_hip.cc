#include "hip/hip_runtime.h"
#include "caffe2/operators/space_batch_op.h"
#include "caffe2/core/hip/common_hip.h"
#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

__global__ void SpaceToBatch(
    int N,
    int output_batch,
    int output_depth,
    int output_height,
    int output_width,
    int input_batch,
    int input_depth,
    int input_height,
    int input_width,
    const int pad_l,
    const int pad_t,
    int block_size,
    const float* input,
    float* output) {
  HIP_1D_KERNEL_LOOP(i, N) {
    // Recall:
    // const auto output_offset =
    //     ((out_b * output_depth + d) * output_height + out_h) * output_width +
    //     out_w;
    const int out_w = i % output_width;
    const int i_2 = i / output_width;
    const int out_h = i_2 % output_height;
    const int i_3 = i_2 / output_height;
    const int d = i_3 % output_depth;
    const int out_b = i_3 / output_depth;

    const int in_b = out_b % input_batch;
    const int offset_w = (out_b / input_batch) % block_size;
    const int offset_h = (out_b / input_batch) / block_size;
    const int in_h = out_h * block_size + offset_h - pad_t;
    const int in_w = out_w * block_size + offset_w - pad_l;

    if (in_h >= 0 && in_w >= 0 && in_h < input_height && in_w < input_width) {
      const auto input_offset =
          ((in_b * input_depth + d) * input_height + in_h) * input_width +
          in_w;
      output[i] = input[input_offset];
    } else {
      output[i] = 0.0;
    }
  }
}

template <>
void spaceToBatch<HIPContext>(
    const Tensor& input,
    int pad_t,
    int pad_l,
    int block_size,
    Tensor* output,
    HIPContext* context) {
  const int output_batch = output->dim32(0);
  const int output_depth = output->dim32(1);
  const int output_height = output->dim32(2);
  const int output_width = output->dim32(3);

  const int input_batch = input.dim32(0);
  const int input_depth = input.dim32(1);
  const int input_height = input.dim32(2);
  const int input_width = input.dim32(3);
  const int N = output->size();
 hipLaunchKernelGGL( SpaceToBatch, 
      dim3(CAFFE_GET_BLOCKS(N)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(), 
      static_cast<int>(N),
      static_cast<int>(output_batch),
      static_cast<int>(output_depth),
      static_cast<int>(output_height),
      static_cast<int>(output_width),
      static_cast<int>(input_batch),
      static_cast<int>(input_depth),
      static_cast<int>(input_height),
      static_cast<int>(input_width),
      static_cast<const int>(pad_l),
      static_cast<const int>(pad_t),
      static_cast<int>(block_size),
      input.data<float>(),
      output->template mutable_data<float>());
}


__global__ void BatchToSpace(
    int N,
    int output_batch,
    int output_depth,
    int output_height,
    int output_width,
    int input_batch,
    int input_depth,
    int input_height,
    int input_width,
    const int pad_l,
    const int pad_t,
    int block_size,
    const float* input,
    float* output) {
  HIP_1D_KERNEL_LOOP(i, N) {
    // Recall:
    // const auto input_offset = ((in_b * input_depth + d) *
    //   input_height + in_h) * input_width + in_w;
    const int in_w = i  % input_width;
    const int i_2 = i / input_width;
    const int in_h = i_2 % input_height;
    const int i_3 = i_2 / input_height;
    const int d = i_3 % input_depth;
    const int in_b = i_3 / input_depth;

    const int out_b = in_b % output_batch;
    const int offset_w = (in_b / output_batch) % block_size;
    const int offset_h = (in_b / output_batch) / block_size;
    const int out_h = in_h * block_size + offset_h - pad_t;
    const int out_w = in_w * block_size + offset_w - pad_l;

    if (out_h >= 0 && out_w >= 0 && out_h < output_height &&
        out_w < output_width) {
      const auto output_offset =
          ((out_b * output_depth + d) * output_height + out_h) *
          output_width +
          out_w;
      output[output_offset] = input[i];
    }
  }
}

template <>
void batchToSpace(
    const Tensor& input,
    int pad_t,
    int pad_l,
    int block_size,
    Tensor* output,
    HIPContext* context) {
  CAFFE_ENFORCE(input.ndim() == 4);
  CAFFE_ENFORCE(output->ndim() == 4);

  const int output_batch = output->dim32(0);
  const int output_depth = output->dim32(1);
  const int output_height = output->dim32(2);
  const int output_width = output->dim32(3);

  const int input_batch = input.dim32(0);
  const int input_depth = input.dim32(1);
  const int input_height = input.dim32(2);
  const int input_width = input.dim32(3);
  const int N = input.size();
 hipLaunchKernelGGL( BatchToSpace, 
      dim3(CAFFE_GET_BLOCKS(N)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(), 
      static_cast<int>(N),
      static_cast<int>(output_batch),
      static_cast<int>(output_depth),
      static_cast<int>(output_height),
      static_cast<int>(output_width),
      static_cast<int>(input_batch),
      static_cast<int>(input_depth),
      static_cast<int>(input_height),
      static_cast<int>(input_width),
      static_cast<const int>(pad_l),
      static_cast<const int>(pad_t),
      static_cast<int>(block_size),
      input.data<float>(),
      output->template mutable_data<float>());
}

REGISTER_HIP_OPERATOR(SpaceToBatch, SpaceToBatchOp<HIPContext>);
REGISTER_HIP_OPERATOR(BatchToSpace, BatchToSpaceOp<HIPContext>);

}
