#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/copy_op.h"

namespace caffe2 {

template <>
class CopyOnDeviceLikeOp<HIPContext, HIPContext, HIPContext>
    : public Operator<HIPContext> {
 public:
  CopyOnDeviceLikeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<HIPContext>(operator_def, ws) {}
  USE_OPERATOR_FUNCTIONS(HIPContext);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = OperatorBase::Output<Tensor>(0, HIP);
    HIPContext context(GetGPUIDForPointer(Input(1).raw_data()));
    output->ResizeLike(input);
    context.template CopyItems<HIPContext, HIPContext>(
        input.meta(),
        input.size(),
        input.raw_data(),
        output->raw_mutable_data(input.meta()));
    return true;
  }
};

// From CPU, copy it to whatever the current context
REGISTER_HIP_OPERATOR(
    CopyFromCPUInput,
    CopyOp<HIPContext, HIPContext, CPUContext>);

// CopyGPUToCPU and CopyCPUToGPU should both be carried out in a cuda context,
// since gpu code will be involved.
REGISTER_HIP_OPERATOR(
    CopyGPUToCPU,
    CopyOp<HIPContext, CPUContext, HIPContext>);
REGISTER_HIP_OPERATOR(
    CopyCPUToGPU,
    CopyOp<HIPContext, HIPContext, CPUContext>);
// If we only specify Copy, we assume that it is a gpu to gpu copy - maybe
// involving different GPUs.
REGISTER_HIP_OPERATOR(Copy, CopyOp<HIPContext, HIPContext, HIPContext>);

REGISTER_HIP_OPERATOR(
    CopyOnDeviceLike,
    CopyOnDeviceLikeOp<HIPContext, HIPContext, HIPContext>);
} // namespace caffe2
