#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/mean_op.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(Mean, MeanOp<HIPContext>);
REGISTER_HIP_OPERATOR(MeanGradient, MeanGradientOp<HIPContext>);

} // namespace caffe2
