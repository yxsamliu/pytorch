#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/loss_op.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(AveragedLoss, AveragedLoss<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    AveragedLossGradient,
    AveragedLossGradient<float, HIPContext>);
}  // namespace caffe2
