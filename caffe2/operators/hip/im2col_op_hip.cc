#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/im2col_op.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(Im2Col, Im2ColOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(Col2Im, Col2ImOp<float, HIPContext>);

} // namespace caffe2
