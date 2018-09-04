#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/conv_transpose_op.h"
#include "caffe2/operators/conv_transpose_op_impl.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(ConvTranspose, ConvTransposeOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    ConvTransposeGradient,
    ConvTransposeGradientOp<float, HIPContext>);
} // namespace caffe2
