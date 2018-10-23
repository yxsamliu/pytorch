#include "caffe2/core/hip/context_hip.h"
#include "caffe2/sgd/iter_op.h"

namespace caffe2 {

REGISTER_HIP_OPERATOR(Iter, IterOp<HIPContext>);
REGISTER_HIP_OPERATOR(AtomicIter, AtomicIterOp<HIPContext>);

} // namespace caffe2
