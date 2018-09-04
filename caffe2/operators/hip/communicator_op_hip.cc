#include "caffe2/core/hip/context_hip.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/no_default_engine_op.h"

namespace caffe2 {
// Communication operators do not have default engines.
REGISTER_HIP_OPERATOR(CreateCommonWorld, NoDefaultEngineOp<HIPContext>);
REGISTER_HIP_OPERATOR(CloneCommonWorld, NoDefaultEngineOp<HIPContext>);
REGISTER_HIP_OPERATOR(Broadcast, NoDefaultEngineOp<HIPContext>);
REGISTER_HIP_OPERATOR(Reduce, NoDefaultEngineOp<HIPContext>);
REGISTER_HIP_OPERATOR(Allgather, NoDefaultEngineOp<HIPContext>);
REGISTER_HIP_OPERATOR(Allreduce, NoDefaultEngineOp<HIPContext>);
REGISTER_HIP_OPERATOR(SendTensor, NoDefaultEngineOp<HIPContext>);
REGISTER_HIP_OPERATOR(ReceiveTensor, NoDefaultEngineOp<HIPContext>);

} // namespace caffe2
