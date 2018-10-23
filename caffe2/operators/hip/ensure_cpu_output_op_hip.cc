#include "caffe2/operators/ensure_cpu_output_op.h"

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {
// From HIP Context, takes either HIP or CPU tensor as input, and produce
// TensorCPU
REGISTER_HIP_OPERATOR(EnsureCPUOutput, EnsureCPUOutputOp<HIPContext>);
} // namespace caffe2
