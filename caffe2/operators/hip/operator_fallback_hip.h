#ifndef CAFFE2_OPERATORS_OPERATOR_FALLBACK_H_
#define CAFFE2_OPERATORS_OPERATOR_FALLBACK_H_

#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {

/**
 * @brief A templated class to allow one to wrap a CPU operator as a HIP
 * operator.
 *
 * This class can be used when one does not have the HIP implementation ready
 * yet for an operator. Essentially, what this op does is to automatically
 * deal with data copy for you. Plausibly, this causes a lot of overhead and
 * is not optimal, so you should use this operator mostly for quick prototyping
 * purpose.
 *
 * All the input and output of the original operator should be TensorCPU.
 *
 * Example usage: if you have a class MyMagicOp that is CPU based, and you use
 * the registration code
 *     REGISTER_CPU_OPERATOR(MyMagic, MyMagicOp);
 * to register the CPU side, you can create its corresponding GPU operator
 * (with performance hits of course) via
 *     REGISTER_HIP_OPERATOR(MyMagic,
 *                            GPUFallbackOp);
 * Note that you will need to make sure that the operators actually share the
 * same name.
 *
 * Advanced usage: if you want to have some specific outputs never copied, you
 * can use the SkipOutputCopy template argument to do that. For example, if
 * MyMagic produces two outputs and the first output is always going to live on
 * the CPU, you can do
 *     REGISTER_HIP_OPERATOR(MyMagic,
 *                            GPUFallbackOpEx<SkipIndices<0>>);
 */
template <typename SkipOutputCopy>
class GPUFallbackOpEx final : public Operator<HIPContext> {
 public:
  USE_OPERATOR_FUNCTIONS(HIPContext);
  GPUFallbackOpEx(const OperatorDef& def, Workspace* ws)
      : Operator<HIPContext>(def, ws) {
    CAFFE_ENFORCE_EQ(def.device_option().device_type(), PROTO_HIP);
    OperatorDef base_def_(def);
    // base_def_ runs on CPU, so we will set its device option to CPU.
    base_def_.clear_device_option();
    base_def_.mutable_device_option()->set_device_type(PROTO_CPU);
    // Set up the symbols for the local workspace.
    for (const string& name : def.input()) {
      local_input_blobs_.push_back(local_ws_.CreateBlob(name));
      CHECK_NOTNULL(local_input_blobs_.back());
    }
    base_op_ = CreateOperator(base_def_, &local_ws_);
    for (const string& name : def.output()) {
      local_output_blobs_.push_back(local_ws_.GetBlob(name));
      CHECK_NOTNULL(local_output_blobs_.back());
    }
  }

  bool RunOnDevice() override {
    bool need_sync = false;
    for (int i = 0; i < InputSize(); ++i) {
      if (this->template InputIsType<Tensor>(i, HIP)) {
        local_input_blobs_[i]->GetMutableTensor(CPU)->CopyFrom(
            Input(i), &context_);
        need_sync = true;
      } else {
        VLOG(1) << "Input " << i << " is not TensorHIP. Skipping copy.";
        // Note(jiayq): This removes a const but conceptually
        // local_input_blobs will only be used as const blob input for the
        // base op so we are still fine.
        local_input_blobs_[i]->ShareExternal(
            const_cast<void*>(OperatorBase::Inputs()[i]->GetRaw()),
            OperatorBase::Inputs()[i]->meta());
      }
    }

    // Sync to make sure copies are done.
    if (need_sync) {
      context_.FinishDeviceComputation();
    }

    if (!base_op_->Run()) {
      LOG(ERROR) << "Base op run failed in GPUFallbackOp. Def: "
                 << ProtoDebugString(this->debug_def());
      return false;
    }
    for (int i = 0; i < OutputSize(); ++i) {
      if (SkipOutputCopy::Contains(i)) {
        VLOG(1) << "Copy output: index " << i << " skipped.";
        continue;
      }
      CAFFE_ENFORCE(
          local_output_blobs_[i]->template IsType<Tensor>(CPU),
          "GPU fallback op currently does not support non-TensorCPU "
          "output type who needs copying.");
      Output(i)->CopyFrom(local_output_blobs_[i]->template Get<TensorCPU>());
    }
    return true;
  }

 protected:
  Workspace local_ws_;
  vector<Blob*> local_input_blobs_;
  vector<Blob*> local_output_blobs_;
  unique_ptr<OperatorBase> base_op_;
};

using GPUFallbackOp = GPUFallbackOpEx<SkipIndices<>>;

} // namespace caffe2

#endif // CAFFE2_OPERATORS_OPERATOR_FALLBACK_H_
