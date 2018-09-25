#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/load_save_op.h"

namespace caffe2 {

template <>
void LoadOp<HIPContext>::SetCurrentDevice(BlobProto* proto) {
  if (proto->has_tensor()) {
    auto* device_detail = proto->mutable_tensor()->mutable_device_detail();
    device_detail->set_device_type(PROTO_HIP);
    device_detail->set_hip_gpu_id(CaffeHipGetDevice());
  }
}

REGISTER_HIP_OPERATOR(Load, LoadOp<HIPContext>);
REGISTER_HIP_OPERATOR(Save, SaveOp<HIPContext>);
REGISTER_HIP_OPERATOR(Checkpoint, CheckpointOp<HIPContext>);
}  // namespace caffe2
