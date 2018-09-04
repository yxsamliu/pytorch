#include "caffe2/operators/enforce_finite_op.h"

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

template <>
template <typename T>
bool EnforceFiniteOp<HIPContext>::DoRunWithType() {
  buffer_.CopyFrom(Input(0), &context_);
  EnforceOnCPU<T>(buffer_);
  return true;
}

REGISTER_HIP_OPERATOR(EnforceFinite, EnforceFiniteOp<HIPContext>);
} // namespace caffe2
