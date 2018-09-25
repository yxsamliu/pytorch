#include "hip/hip_runtime.h"
#include "caffe2/core/hip/common_hip.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/sgd/adadelta_op.h"
#include "caffe2/utils/hip/mixed_utils_hip.h"

namespace caffe2 {

namespace {

__global__ void AdadeltaUpdateKernel(
    int N,
    const float* w,
    const float* g,
    const float* h,
    const float* d,
    const float epsilon,
    const float decay,
    const float* lr,
    float* nw,
    float* nh,
    float* nd) {
  HIP_1D_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float di = d[i];
    float hi = nh[i] = decay * h[i] + (1.0f - decay) * gi * gi;
    float ng = sqrtf(di + epsilon) * rsqrtf(hi + epsilon) * gi;
    nw[i] = w[i] + lr[0] * ng;
    nd[i] = decay * di + (1.0f - decay) * ng * ng;
  }
}

template <>
void AdadeltaUpdate<HIPContext>(
    int N,
    const float* w,
    const float* g,
    const float* h,
    const float* d,
    const float epsilon,
    const float decay,
    const float* lr,
    float* nw,
    float* nh,
    float* nd,
    HIPContext* context) {
 hipLaunchKernelGGL( AdadeltaUpdateKernel, 
      dim3(CAFFE_GET_BLOCKS(N)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context->hip_stream(), static_cast<int>(N), w, g, h, d, epsilon, decay, lr, nw, nh, nd);
}

} // namespace

template <typename SIndex, typename THalf>
__global__ void SparseAdadeltaKernel(
    const size_t N,
    const size_t grad_slice_sz,
    const float epsilon,
    const float decay,
    const SIndex* indices,
    const float* grad,
    const float* lr,
    THalf* param,
    THalf* param_mom,
    THalf* param_mom_delta) {
  const float LR = lr[0];
  HIP_1D_KERNEL_LOOP(i, N) {
    const size_t gradIdx = i;
    const SIndex index = indices[i / grad_slice_sz];
    const size_t paramIdx = index * grad_slice_sz + (i % grad_slice_sz);

    float mom_new = mixed_mult(decay, param_mom[paramIdx]) +
        (1.0f - decay) * grad[gradIdx] * grad[gradIdx];
    mixed_store(&mom_new, &(param_mom[paramIdx]));
    float grad_new = sqrtf(mixed_add(epsilon, param_mom_delta[paramIdx])) *
        rsqrtf(mom_new + epsilon) * grad[gradIdx];
    float param_new = mixed_add(LR * grad_new, param[paramIdx]);
    mixed_store(&param_new, &(param[paramIdx]));
    float mom_delta_new = mixed_mult(decay, param_mom_delta[paramIdx]) +
        (1.0f - decay) * grad_new * grad_new;
    mixed_store(&mom_delta_new, &(param_mom_delta[paramIdx]));
  }
}

template <class Context>
class HIPSparseAdadeltaOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  HIPSparseAdadeltaOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(float, "epsilon", epsilon_, 1e-5f),
        OP_SINGLE_ARG(float, "decay", decay_, 0.95f) {}

  bool RunOnDevice() override {
    // Enforce shapes
    CAFFE_ENFORCE_EQ(Input(PARAM).size(), Input(MOMENT_GRAD).size());
    CAFFE_ENFORCE_EQ(Input(PARAM).size(), Input(MOMENT_DELTA).size());
    CAFFE_ENFORCE_EQ(Input(LR).size(), 1);
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).ndim()));

    // Enforce domain constraints on attributes
    CAFFE_ENFORCE_GE(epsilon_, 0.0f);
    CAFFE_ENFORCE_GT(decay_, 0.0f);
    CAFFE_ENFORCE_LT(decay_, 1.0f);

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename IndexType>
  bool DoRunWithType() {
    auto n = Input(INDICES).size();
    if (n == 0) {
      return true;
    }
    return DispatchHelper<TensorTypes2<float, float16>, IndexType>::call(
        this, Input(PARAM));
  }

  template <typename IndexType, typename THalf>
  bool DoRunWithType2() {
    const auto* lr = Input(LR).template data<float>();
    const auto* indices = Input(INDICES).template data<IndexType>();
    const auto* gradIn = Input(GRAD).template data<float>();
    const auto* paramIn = Input(PARAM).template data<THalf>();
    const auto* momentIn = Input(MOMENT_GRAD).template data<THalf>();
    const auto* momentDeltaIn = Input(MOMENT_DELTA).template data<THalf>();
    auto* paramOut = Output(OUTPUT_PARAM)->template mutable_data<THalf>();
    auto* momentOut =
        Output(OUTPUT_MOMENT_GRAD)->template mutable_data<THalf>();
    auto* momentDeltaOut =
        Output(OUTPUT_MOMENT_DELTA)->template mutable_data<THalf>();

    auto N = Input(GRAD).size();
    auto grad_slice_sz = Input(GRAD).size_from_dim(Input(INDICES).ndim());
    if (N == 0) {
      // empty grad, nothing to do here, not even launching the kernel
      return true;
    }
   hipLaunchKernelGGL( SparseAdadeltaKernel<IndexType, THalf>
        , dim3(CAFFE_GET_BLOCKS(N)),
           dim3(CAFFE_HIP_NUM_THREADS),
           0,
           context_.hip_stream(), 
            N,
            grad_slice_sz,
            epsilon_,
            decay_,
            indices,
            gradIn,
            lr,
            paramOut,
            momentOut,
            momentDeltaOut);
    return true;
  }

 protected:
  const float epsilon_;
  const float decay_;
  INPUT_TAGS(PARAM, MOMENT_GRAD, MOMENT_DELTA, INDICES, GRAD, LR);
  OUTPUT_TAGS(OUTPUT_PARAM, OUTPUT_MOMENT_GRAD, OUTPUT_MOMENT_DELTA);
};

REGISTER_HIP_OPERATOR(Adadelta, AdadeltaOp<HIPContext>);
REGISTER_HIP_OPERATOR(SparseAdadelta, HIPSparseAdadeltaOp<HIPContext>);
} // namespace caffe2
