#include "caffe2/operators/exp_op.h"

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(
    Exp,
    UnaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        ExpFunctor<HIPContext>>);

} // namespace caffe2
