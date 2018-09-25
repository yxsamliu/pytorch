#include "caffe2/core/hip/common_hip.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/tensor_protos_db_input.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(TensorProtosDBInput, TensorProtosDBInput<HIPContext>);
}  // namespace caffe2
