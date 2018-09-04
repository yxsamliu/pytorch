#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/transpose_op.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(Transpose, TransposeOp<HIPContext>);

} // namespace caffe2
