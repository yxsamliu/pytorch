#include "caffe2/operators/expand_op.h"

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(
    Expand,
    ExpandOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        HIPContext>);
REGISTER_HIP_OPERATOR(
    ExpandGradient,
    ExpandGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        HIPContext>);
} // namespace caffe2
