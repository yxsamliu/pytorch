#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/given_tensor_byte_string_to_uint8_fill_op.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(
    GivenTensorByteStringToUInt8Fill,
    GivenTensorByteStringToUInt8FillOp<HIPContext>);
}
