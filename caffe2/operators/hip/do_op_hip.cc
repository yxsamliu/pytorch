#include "caffe2/operators/do_op.h"

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(Do, DoOp<HIPContext>);

} // namespace caffe2
