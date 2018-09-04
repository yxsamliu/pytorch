#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/batch_matmul_op.h"

namespace caffe2 {
namespace {

class BatchMatMulOpGPUTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!HasHipGPU()) {
      return;
    }
    option_.set_device_type(HIP);
    hip_context_ = make_unique<HIPContext>(option_);
    def_.set_name("test");
    def_.set_type("BatchMatMul");
    def_.add_input("A");
    def_.add_input("B");
    def_.add_output("Y");
    def_.mutable_device_option()->set_device_type(HIP);
  }

  void AddConstInput(
      const std::vector<TIndex>& dims,
      const float value,
      const string& name) {
    Blob* blob = ws_.CreateBlob(name);
    auto* tensor = blob->GetMutableTensor(HIP);
    tensor->Resize(dims);
    math::Set<float, HIPContext>(
        tensor->size(),
        value,
        tensor->template mutable_data<float>(),
        hip_context_.get());
  }

  void VerifyOutput(const std::vector<TIndex>& dims, const float value) const {
    const Blob* Y_blob = ws_.GetBlob("Y");
    ASSERT_NE(nullptr, Y_blob);
    const auto& Y = Y_blob->Get<Tensor>();
    Tensor Y_cpu(Y, CPU);
    const auto& Y_dims = Y_cpu.dims();
    ASSERT_EQ(dims.size(), Y_dims.size());
    for (std::size_t i = 0; i < dims.size(); ++i) {
      ASSERT_EQ(dims[i], Y_dims[i]);
    }
    for (int i = 0; i < Y_cpu.size(); ++i) {
      EXPECT_FLOAT_EQ(value, Y_cpu.data<float>()[i]);
    }
  }

  DeviceOption option_;
  std::unique_ptr<HIPContext> hip_context_;
  Workspace ws_;
  OperatorDef def_;
};

TEST_F(BatchMatMulOpGPUTest, BatchMatMulOpGPUNormalTest) {
  if (!HasHipGPU()) {
    return;
  }
  AddConstInput(std::vector<TIndex>{3, 5, 10}, 1.0f, "A");
  AddConstInput(std::vector<TIndex>{3, 10, 6}, 1.0f, "B");
  std::unique_ptr<OperatorBase> op(CreateOperator(def_, &ws_));
  ASSERT_NE(nullptr, op);
  ASSERT_TRUE(op->Run());
  VerifyOutput(std::vector<TIndex>{3, 5, 6}, 10.0f);
}

TEST_F(BatchMatMulOpGPUTest, BatchMatMulOpGPUBroadcastTest) {
  if (!HasHipGPU()) {
    return;
  }
  auto* arg = def_.add_arg();
  arg->set_name("broadcast");
  arg->set_i(1);
  AddConstInput(std::vector<TIndex>{3, 5, 10}, 1.0f, "A");
  AddConstInput(std::vector<TIndex>{2, 3, 10, 6}, 1.0f, "B");
  std::unique_ptr<OperatorBase> op(CreateOperator(def_, &ws_));
  ASSERT_NE(nullptr, op);
  ASSERT_TRUE(op->Run());
  VerifyOutput(std::vector<TIndex>{2, 3, 5, 6}, 10.0f);
}

} // namespace
} // namespace caffe2
