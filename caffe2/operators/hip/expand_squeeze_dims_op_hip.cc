#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/expand_squeeze_dims_op.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(Squeeze, SqueezeOp<HIPContext>);
REGISTER_HIP_OPERATOR(ExpandDims, ExpandDimsOp<HIPContext>);
} // namespace caffe2
