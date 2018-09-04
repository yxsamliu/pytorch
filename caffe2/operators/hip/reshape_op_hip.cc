#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/reshape_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(Reshape, ReshapeOp<float, HIPContext>);

} // namespace caffe2
