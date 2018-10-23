#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/assert_op.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(Assert, AssertOp<HIPContext>);

} // namespace caffe2
