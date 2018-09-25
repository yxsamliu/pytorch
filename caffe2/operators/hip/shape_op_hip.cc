#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/shape_op.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(Shape, ShapeOp<HIPContext>);
}
