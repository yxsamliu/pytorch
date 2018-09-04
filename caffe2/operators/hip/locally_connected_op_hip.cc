#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/locally_connected_op.h"
#include "caffe2/operators/locally_connected_op_impl.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(LC, LocallyConnectedOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    LCGradient,
    LocallyConnectedGradientOp<float, HIPContext>);

REGISTER_HIP_OPERATOR(LC1D, LocallyConnectedOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    LC1DGradient,
    LocallyConnectedGradientOp<float, HIPContext>);

REGISTER_HIP_OPERATOR(LC2D, LocallyConnectedOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    LC2DGradient,
    LocallyConnectedGradientOp<float, HIPContext>);

REGISTER_HIP_OPERATOR(LC3D, LocallyConnectedOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    LC3DGradient,
    LocallyConnectedGradientOp<float, HIPContext>);

} // namespace caffe2
