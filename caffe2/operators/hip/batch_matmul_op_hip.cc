#include "caffe2/operators/batch_matmul_op.h"

#include "caffe2/core/hip/context_hip.h"

namespace caffe2 {

template <>
bool BatchMatMulOp<HIPContext, DefaultEngine>::RunOnDevice() {
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
}

REGISTER_HIP_OPERATOR(BatchMatMul, BatchMatMulOp<HIPContext>);

#if HIP_VERSION >= 9000

template <>
bool BatchMatMulOp<HIPContext, TensorCoreEngine>::RunOnDevice() {
    return DispatchHelper<TensorTypes<float, at::Half>>::call(this, Input(0));
}

REGISTER_HIP_OPERATOR_WITH_ENGINE(
    BatchMatMul,
    TENSORCORE,
    BatchMatMulOp<HIPContext, TensorCoreEngine>);
#endif

} // namespace caffe2
