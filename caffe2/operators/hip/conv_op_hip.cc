#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_op_impl.h"
#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(Conv, ConvOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(ConvGradient, ConvGradientOp<float, HIPContext>);

REGISTER_HIP_OPERATOR(Conv1D, ConvOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(Conv1DGradient, ConvGradientOp<float, HIPContext>);

REGISTER_HIP_OPERATOR(Conv2D, ConvOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(Conv2DGradient, ConvGradientOp<float, HIPContext>);

REGISTER_HIP_OPERATOR(Conv3D, ConvOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(Conv3DGradient, ConvGradientOp<float, HIPContext>);
}  // namespace caffe2
