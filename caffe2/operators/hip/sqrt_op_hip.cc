#include "caffe2/operators/sqrt_op.h"

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(
    Sqrt,
    UnaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        SqrtFunctor<HIPContext>>);

} // namespace caffe2
