#include "caffe2/operators/log_op.h"

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(
    Log,
    UnaryElementwiseOp<
        TensorTypes<float>,
        HIPContext,
        LogFunctor<HIPContext>>);

} // namespace caffe2
