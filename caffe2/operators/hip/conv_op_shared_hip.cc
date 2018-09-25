#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/conv_op_shared.h"

namespace caffe2 {

template <>
void createSharedBuffer<HIPContext>(Workspace* ws) {
  auto* mutexPtr = ws->CreateBlob("__CAFFE2_SHARED_CONV_BUFFER_HIP_MUTEX__")
                       ->GetMutable<std::unique_ptr<std::mutex>>();
  mutexPtr->reset(new std::mutex());
  ws->CreateBlob("__CAFFE2_SHARED_CONV_BUFFER_HIP__");
}

template <>
void runWithSharedBuffer<HIPContext>(
    Workspace* ws,
    std::function<void(Tensor* buffer)> f) {
  auto* mutexBlob = ws->GetBlob("__CAFFE2_SHARED_CONV_BUFFER_HIP_MUTEX__");
  CAFFE_ENFORCE(mutexBlob, "Must call createSharedBuffer() first");

  auto* mutexPtr = mutexBlob->GetMutable<std::unique_ptr<std::mutex>>();
  std::lock_guard<std::mutex> g(**mutexPtr);
  auto* buffer =
      ws->GetBlob("__CAFFE2_SHARED_CONV_BUFFER_HIP__")->GetMutableTensor(HIP);
  f(buffer);
}
}
