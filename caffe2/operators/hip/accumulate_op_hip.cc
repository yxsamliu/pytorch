#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/accumulate_op.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(Accumulate, AccumulateOp<float, HIPContext>);
}  // namespace caffe2
