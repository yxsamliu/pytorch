#include "caffe2/operators/if_op.h"

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(If, IfOp<HIPContext>);

} // namespace caffe2
