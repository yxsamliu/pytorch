#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/negate_gradient_op.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(NegateGradient, NegateGradientOp<HIPContext>)
} // namespace caffe2
