#include "caffe2/core/hip/common_hip.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/fully_connected_op.h"

namespace caffe2 {

namespace {

constexpr int kFp16HIPDevicePropMajor = 6;

template <class FullyConnectedOp>
bool RunFullyConnectedOpOnHIPDevice(
    const bool float16_compute,
    FullyConnectedOp* op) {
  if (op->Input(0).template IsType<float>()) {
    return op->template DoRunWithType<
        float, // X
        float, // W
        float, // B
        float, // Y
        float>(); // Math
  } else if (op->Input(0).template IsType<float16>()) {
    if (float16_compute) {
      const hipDeviceProp_t& prop = GetDeviceProperty(0);
      if (prop.major >= kFp16HIPDevicePropMajor) {
        return op->template DoRunWithType<
            float16, // X
            float16, // W
            float16, // B
            float16, // Y
            float16>(); // Math
      } else {
        LOG(INFO) << "HIP Device does not support FP16 computation, "
                     "falling back to FP32.";
        return op->template DoRunWithType<
            float16, // X
            float16, // W
            float16, // B
            float16, // Y
            float>(); // Math
      }
    } else {
      return op->template DoRunWithType<
          float16, // X
          float16, // W
          float16, // B
          float16, // Y
          float>(); // Math
    }
  } else {
    CAFFE_THROW("Unsupported type");
  }
  return false;
}

template <class FullyConnectedGradientOp>
bool RunFullyConnectedGradientOpOnHIPDevice(
    const bool float16_compute,
    FullyConnectedGradientOp* op) {
  if (op->Input(0).template IsType<float>()) {
    return op->template DoRunWithType<
        float, //  X
        float, //  W
        float, // dY
        float, //  B
        float, // dX
        float, // dW
        float, // dB
        float>(); // Math
  } else if (op->Input(0).template IsType<float16>()) {
    if (float16_compute) {
      const hipDeviceProp_t& prop = GetDeviceProperty(0);
      if (prop.major >= kFp16HIPDevicePropMajor) {
        return op->template DoRunWithType<
            float16, //  X
            float16, //  W
            float16, // dY
            float16, //  B
            float16, // dX
            float16, // dW
            float16, // dB
            float16>(); // Math
      } else {
        LOG(INFO) << "HIP Device does not support FP16 computation, "
                     "falling back to FP32.";
        return op->template DoRunWithType<
            float16, //  X
            float16, //  W
            float16, // dY
            float16, //  B
            float16, // dX
            float16, // dW
            float16, // dB
            float>(); // Math
      }
    } else {
      return op->template DoRunWithType<
          float16, //  X
          float16, //  W
          float16, // dY
          float16, //  B
          float16, // dX
          float16, // dW
          float16, // dB
          float>(); // Math
    }
  } else {
    CAFFE_THROW("Unsupported type");
  }
  return false;
}

} // namespace

// The RunFullyConnectedOpOnHIPDevice Function will use the pointer of current
// op and the DoRunWithType will make sure to run the correct things.
template <>
bool FullyConnectedOp<HIPContext>::RunOnDevice() {
  return RunFullyConnectedOpOnHIPDevice(float16_compute_, this);
}

template <>
bool FullyConnectedOp<
    HIPContext,
    DefaultEngine,
    false /* don't transpose weight */>::RunOnDevice() {
  return RunFullyConnectedOpOnHIPDevice(float16_compute_, this);
}

template <>
bool FullyConnectedGradientOp<HIPContext>::RunOnDevice() {
  return RunFullyConnectedGradientOpOnHIPDevice(float16_compute_, this);
}

template <>
bool FullyConnectedGradientOp<
    HIPContext,
    DefaultEngine,
    false /* don't transpose weight */>::RunOnDevice() {
  return RunFullyConnectedGradientOpOnHIPDevice(float16_compute_, this);
}

#if HIP_VERSION >= 9000

// Require these to be defined otherwise TensorCore FC ops will end
// up calling the default FC implementation which doesn't have
// fp16 support...

template <>
bool FullyConnectedOp<HIPContext, TensorCoreEngine>::RunOnDevice() {
  return RunFullyConnectedOpOnHIPDevice(false /* float16_compute */, this);
}

template <>
bool FullyConnectedOp<
    HIPContext,
    TensorCoreEngine,
    false /* don't transpose weight */>::RunOnDevice() {
  return RunFullyConnectedOpOnHIPDevice(false /* float16_compute */, this);
}

template <>
bool FullyConnectedGradientOp<HIPContext, TensorCoreEngine>::RunOnDevice() {
  return RunFullyConnectedGradientOpOnHIPDevice(
      false /* float16_compute */, this);
}

template <>
bool FullyConnectedGradientOp<
    HIPContext,
    TensorCoreEngine,
    false /* don't transpose weight */>::RunOnDevice() {
  return RunFullyConnectedGradientOpOnHIPDevice(
      false /* float16_compute */, this);
}

#endif

REGISTER_HIP_OPERATOR(FC, FullyConnectedOp<HIPContext>);
REGISTER_HIP_OPERATOR(FCGradient, FullyConnectedGradientOp<HIPContext>);

REGISTER_HIP_OPERATOR(
    FCTransposed,
    FullyConnectedOp<
        HIPContext,
        DefaultEngine,
        false /* don't transpose weight */>);
REGISTER_HIP_OPERATOR(
    FCTransposedGradient,
    FullyConnectedGradientOp<
        HIPContext,
        DefaultEngine,
        false /* don't transpose weight */>);

#if HIP_VERSION >= 9000
REGISTER_HIP_OPERATOR_WITH_ENGINE(
    FC,
    TENSORCORE,
    FullyConnectedOp<HIPContext, TensorCoreEngine>);
REGISTER_HIP_OPERATOR_WITH_ENGINE(
    FCGradient,
    TENSORCORE,
    FullyConnectedGradientOp<HIPContext, TensorCoreEngine>);

REGISTER_HIP_OPERATOR_WITH_ENGINE(
    FCTransposed,
    TENSORCORE,
    FullyConnectedOp<
        HIPContext,
        TensorCoreEngine,
        false /* don't transpose weight */>);
REGISTER_HIP_OPERATOR_WITH_ENGINE(
    FCTransposedGradient,
    TENSORCORE,
    FullyConnectedGradientOp<
        HIPContext,
        TensorCoreEngine,
        false /* don't transpose weight */>);
#endif

} // namespace caffe2
