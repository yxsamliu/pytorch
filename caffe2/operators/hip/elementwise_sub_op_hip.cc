#include "caffe2/operators/elementwise_sub_op.h"

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(
    Sub,
    BinaryElementwiseOp<NumericTypes, HIPContext, SubFunctor<HIPContext>>);
REGISTER_HIP_OPERATOR(
    SubGradient,
    BinaryElementwiseGradientOp<
        NumericTypes,
        HIPContext,
        SubFunctor<HIPContext>>);

} // namespace caffe2
