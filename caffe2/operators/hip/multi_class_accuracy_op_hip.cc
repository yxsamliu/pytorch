#include "hip/hip_runtime.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/multi_class_accuracy_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {
__global__ void MultiClassAccuracyKernel(const int N, const int D, const float* Xdata,
    const int* labeldata, float* accuracies, int* amounts) {
  HIP_1D_KERNEL_LOOP(i, N) {
    float maxval = Xdata[i * D];
    int maxid = 0;
    for (int j = 1; j < D; ++j) {
      if (Xdata[i * D + j] > maxval) {
        maxval = Xdata[i * D + j];
        maxid = j;
      }
    }
    int labelid = labeldata[i];
    if (maxid == labelid) {
      atomicAdd(accuracies + labelid, static_cast<float>(1));
    }
    atomicAdd(amounts + labelid, static_cast<int>(1));
  }
}
__global__ void MultiClassAccuracyDivideKernel(
  const int D, float* accuracies, const int* amounts) {
  HIP_1D_KERNEL_LOOP(i, D) {
    if (amounts[i]) {
      accuracies[i] /= amounts[i];
    }
  }
}
}  // namespace

template <>
bool MultiClassAccuracyOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(PREDICTION);
  auto& label = Input(LABEL);
  auto* Y0 = Output(0);
  auto* Y1 = Output(1);
  DCHECK_EQ(X.ndim(), 2);
  // amount, number of instances
  int N = X.dim32(0);
  // dimension, number of classes
  int D = X.dim32(1);
  DCHECK_EQ(label.ndim(), 1);
  DCHECK_EQ(label.dim32(0), N);
  Y0->Resize(D);
  Y1->Resize(D);

  const float* Xdata = X.data<float>();
  const int* labeldata = label.data<int>();
  float* accuracies = Y0->template mutable_data<float>();
  int* amounts = Y1->template mutable_data<int>();
  math::Set<float, HIPContext>(D, 0.0, accuracies, &context_);
  math::Set<int, HIPContext>(D, 0, amounts, &context_);

 hipLaunchKernelGGL( MultiClassAccuracyKernel, dim3(CAFFE_GET_BLOCKS(N)), dim3(CAFFE_HIP_NUM_THREADS),
                              0, context_.hip_stream(), 
      static_cast<const int>(N), static_cast<const int>(D), Xdata, labeldata, accuracies, amounts);
 hipLaunchKernelGGL( MultiClassAccuracyDivideKernel, dim3(CAFFE_GET_BLOCKS(D)), dim3(CAFFE_HIP_NUM_THREADS),
                                  0, context_.hip_stream(), 
    static_cast<const int>(D), accuracies, amounts);
  return true;
}

REGISTER_HIP_OPERATOR(
  MultiClassAccuracy, MultiClassAccuracyOp<float, HIPContext>);
}  // namespace caffe2
