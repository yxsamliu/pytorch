#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/concat_split_op.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(Split, SplitOp<HIPContext>);
REGISTER_HIP_OPERATOR(Concat, ConcatOp<HIPContext>);

// Backward compatibility settings
REGISTER_HIP_OPERATOR(DepthSplit, SplitOp<HIPContext>);
REGISTER_HIP_OPERATOR(DepthConcat, ConcatOp<HIPContext>);

REGISTER_HIP_OPERATOR(SplitByLengths, SplitByLengthsOp<HIPContext>);
} // namespace caffe2
