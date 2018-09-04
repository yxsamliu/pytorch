#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/counter_ops.h"

namespace caffe2 {
REGISTER_HIP_OPERATOR(CreateCounter, CreateCounterOp<int64_t, HIPContext>);
REGISTER_HIP_OPERATOR(ResetCounter, ResetCounterOp<int64_t, HIPContext>);
REGISTER_HIP_OPERATOR(CountDown, CountDownOp<int64_t, HIPContext>);
REGISTER_HIP_OPERATOR(
    CheckCounterDone,
    CheckCounterDoneOp<int64_t, HIPContext>);
REGISTER_HIP_OPERATOR(CountUp, CountUpOp<int64_t, HIPContext>);
REGISTER_HIP_OPERATOR(RetrieveCount, RetrieveCountOp<int64_t, HIPContext>);
} // namespace caffe2
