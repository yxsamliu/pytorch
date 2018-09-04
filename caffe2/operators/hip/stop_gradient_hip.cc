#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/stop_gradient.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(StopGradient, StopGradientOp<HIPContext>);
}  // namespace caffe2
