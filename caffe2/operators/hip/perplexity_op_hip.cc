#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/perplexity_op.h"
#include "caffe2/utils/math.h"
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/system/cuda/execution_policy.h>

namespace caffe2 {

struct perplexity_function
{
  perplexity_function(float p) : pow(p) {}
  __host__ __device__ float operator()(float x) const
  {
    return powf(1.0f/x, pow);
  }
  float pow;
};

template <>
bool PerplexityOp<float, HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);

  DCHECK_EQ(X.ndim(), 1);
  int N = X.dim32(0);

  Y->Resize(vector<TIndex>());
  float* Ydata = Y->template mutable_data<float>();
  const float* Xdata = X.data<float>();

  float perplexity = thrust::transform_reduce(
      #if THRUST_VERSION >= 100800
        thrust::cuda::par.on(context_.hip_stream()),
      #endif  // THRUST_VERSION >= 100800
      Xdata, Xdata + N,
      perplexity_function(1.0f/N),
      1.0f,
      thrust::multiplies<float>());

  math::Set<float, HIPContext>(1, perplexity, Ydata, &context_);
  return true;
}

REGISTER_HIP_OPERATOR(Perplexity, PerplexityOp<float, HIPContext>);
}  // namespace caffe2
