#include "caffe2/operators/negative_op.h"

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(
    Negative,
    UnaryElementwiseOp<
        NumericTypes,
        HIPContext,
        NegativeFunctor<HIPContext>>);

} // namespace caffe2
