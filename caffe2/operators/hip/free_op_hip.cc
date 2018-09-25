#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/free_op.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(Free, FreeOp<HIPContext>);
} // namespace caffe2
