#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/scale_op.h"

namespace caffe2 {

template <>
bool ScaleOp<HIPContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<at::Half, float>>::call(this, Input(0));
}

REGISTER_HIP_OPERATOR(Scale, ScaleOp<HIPContext>);

}  // namespace caffe2
