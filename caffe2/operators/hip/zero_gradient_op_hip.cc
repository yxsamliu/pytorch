#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/zero_gradient_op.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(ZeroGradient, ZeroGradientOp<HIPContext>);
}
