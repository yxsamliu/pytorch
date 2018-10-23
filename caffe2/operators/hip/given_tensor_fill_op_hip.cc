#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/given_tensor_fill_op.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(GivenTensorFill, GivenTensorFillOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(
    GivenTensorDoubleFill,
    GivenTensorFillOp<double, HIPContext>);
REGISTER_HIP_OPERATOR(GivenTensorIntFill, GivenTensorFillOp<int, HIPContext>);
REGISTER_HIP_OPERATOR(
    GivenTensorInt64Fill,
    GivenTensorFillOp<int64_t, HIPContext>);
REGISTER_HIP_OPERATOR(
    GivenTensorBoolFill,
    GivenTensorFillOp<bool, HIPContext>);
}
