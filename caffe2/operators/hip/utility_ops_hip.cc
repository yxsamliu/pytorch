#include "hip/hip_runtime.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/flatten_op.h"
#include "caffe2/operators/minmax_ops.h"
#include "caffe2/operators/utility_ops.h"
#include "caffe2/utils/math.h"

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/unique.h>

namespace caffe2 {

template <>
bool WeightedSumOp<HIPContext>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<float>();
  } else if (Input(0).IsType<at::Half>()) {
    return DoRunWithType<at::Half>();
  } else {
    CAFFE_THROW("Unsupported inputs");
  }
  return false;
}

template <>
bool SumOp<HIPContext>::RunOnDevice() {
  if (Input(0).IsType<float>()) {
    return DoRunWithType<float, float>();
  } else if (Input(0).IsType<at::Half>()) {
    return DoRunWithType<at::Half, at::Half>();
  } else {
    CAFFE_THROW("Unsupported inputs");
  }
  return false;
}

REGISTER_HIP_OPERATOR(Print, PrintOp<HIPContext>);
REGISTER_HIP_OPERATOR(Flatten, FlattenOp<HIPContext>);
REGISTER_HIP_OPERATOR(FlattenToVec, FlattenToVecOp<HIPContext>);
REGISTER_HIP_OPERATOR(Alias, AliasOp<HIPContext>);
REGISTER_HIP_OPERATOR(ResizeLike, ResizeLikeOp<HIPContext>);
REGISTER_HIP_OPERATOR(Sum, SumOp<HIPContext>);
REGISTER_HIP_OPERATOR(WeightedSum, WeightedSumOp<HIPContext>);

REGISTER_HIP_OPERATOR(UnsafeCoalesce, UnsafeCoalesceOp<HIPContext>);

CAFFE_KNOWN_TYPE(const float*);

REGISTER_HIP_OPERATOR(EnsureDense, EnsureDenseOp<HIPContext>);

__global__ void NanCheckKernel(int N, const float* X, bool* result) {
  bool has_nan = false;
  HIP_1D_KERNEL_LOOP(i, N) {
    // Note: we have no need to do early return, since only if this fails
    // will we not need to inspect all elements. No need to optimize the
    // case that will fail.
    has_nan = has_nan || isnan(X[i]) || isinf(X[i]);
  }
  __syncthreads();
  if (has_nan) {
    result[0] = true;
  }
}

template <>
bool NanCheckOp<HIPContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  const size_t N = X.size();
  const float* data_ptr = X.data<float>();

  scratch_.Resize(1);
  math::Set<bool, HIPContext>(
      1, false, scratch_.mutable_data<bool>(), &context_);
 hipLaunchKernelGGL( NanCheckKernel, 
      dim3(CAFFE_GET_BLOCKS(N)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<int>(N), X.data<float>(), scratch_.mutable_data<bool>());

  bool result = false;
  {
    std::lock_guard<std::mutex> lock(HIPContext::mutex());
    HIP_ENFORCE(hipMemcpyAsync(
        &result,
        scratch_.raw_data(),
        1,
        hipMemcpyDefault,
        context_.hip_stream()));
  }
  // Note: we must synchronize here so we can inspect the result
  context_.FinishDeviceComputation();

  // Print out diagnostic info if we have a NaN or inf
  if (result) {
    std::cerr << "Tensor contained NaN or inf: " << this->debug_def().input(0)
              << std::endl;

    for (int j = 0; j < InputSize(); j++) {
      Tensor cpu_X(CPU);
      cpu_X.ResizeLike(Input(j));
      // Hack to cause allocaiton happen here, so it won't happen
      // when we do CopyFrom. We need the mutex then because host->gpu
      // copies seem to possibly lock with NCCL.
      cpu_X.mutable_data<float>();

      {
        std::lock_guard<std::mutex> lock(HIPContext::mutex());
        cpu_X.CopyFrom(Input(j), &context_);
      }
      context_.FinishDeviceComputation();
      std::cerr << "Input tensor: " << j << ": [" << this->debug_def().input(j)
                << "]" << std::endl;
      tensorPrinter_.Print<float>(cpu_X);

      if (j == 0) {
        std::cerr << "NaN idxs:" << std::endl;
        auto* cpu_X_data = cpu_X.data<float>();
        for (size_t i = 0; i < cpu_X.size(); ++i) {
          if (std::isnan(cpu_X_data[i]) || std::isinf(cpu_X_data[i])) {
            std::cerr << i << " ";
          }
        }
      }
      std::cerr << std::endl;
    }
    return false;
  }

  // This op should act as an identity matrix if we don't find any NaNs/infs.
  // Copy over the data if we are not doing this in-place.
  if (&X != Y) {
    Y->CopyFrom(X, &context_);
  }
  return true;
}

REGISTER_HIP_OPERATOR(NanCheck, NanCheckOp<HIPContext>);

__global__ void
ElwiseMaxKernel(const float* X, const float* Y, float* maxout, const int N) {
  HIP_1D_KERNEL_LOOP(i, N) {
    maxout[i] = fmaxf(X[i], Y[i]);
  }
}

template <>
bool MaxOp<float, HIPContext>::Compute() {
  float* output_data = Output(0)->template mutable_data<float>();
  const int N = Input(0).size();

  // Run pairwise-maxes
  for (int i = 1; i < InputSize(); ++i) {
   hipLaunchKernelGGL( ElwiseMaxKernel, 
        dim3(CAFFE_GET_BLOCKS(N)),
        dim3(CAFFE_HIP_NUM_THREADS),
        0,
        context_.hip_stream(), 
        (i == 0 ? Input(0).data<float>() : Output(0)->data<float>()),
        Input(i).data<float>(),
        output_data,
        static_cast<const int>(N));
  }

  return true;
}

REGISTER_HIP_OPERATOR(Max, MaxOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(MaxGradient, MaxGradientOp<float, HIPContext>);

__global__ void
ElwiseMinKernel(const float* X, const float* Y, float* minout, const int N) {
  HIP_1D_KERNEL_LOOP(i, N) {
    minout[i] = fminf(X[i], Y[i]);
  }
}

template <>
bool MinOp<float, HIPContext>::Compute() {
  float* output_data = Output(0)->template mutable_data<float>();
  const int N = Input(0).size();

  // Run pairwise-mines
  for (int i = 1; i < InputSize(); ++i) {
   hipLaunchKernelGGL( ElwiseMinKernel, 
        dim3(CAFFE_GET_BLOCKS(N)),
        dim3(CAFFE_HIP_NUM_THREADS),
        0,
        context_.hip_stream(), 
        (i == 0 ? Input(0).data<float>() : Output(0)->data<float>()),
        Input(i).data<float>(),
        output_data,
        static_cast<const int>(N));
  }

  return true;
}

REGISTER_HIP_OPERATOR(Min, MinOp<float, HIPContext>);
REGISTER_HIP_OPERATOR(MinGradient, MinGradientOp<float, HIPContext>);

template <typename T>
__global__ void
MaxMinGradKernel(int N, const T* mx, const T* x, const T* go, T* gi) {
  HIP_1D_KERNEL_LOOP(i, N) {
    gi[i] = go[i] * (mx[i] == x[i]);
  }
}

template <>
bool SelectGradientOpBase<float, HIPContext>::RunOnDevice() {
  auto& output = Input(0);
  auto& grad_output = Input(1);
  const int kInputStartOffset = 2;

  const float* data = output.data<float>();

  for (int i = 0; i < OutputSize(); i++) {
    auto& input = Input(i + kInputStartOffset);
    auto* grad_input = Output(i);
    grad_input->ResizeLike(input);
   hipLaunchKernelGGL( MaxMinGradKernel, 
        dim3(CAFFE_GET_BLOCKS(input.size())),
        dim3(CAFFE_HIP_NUM_THREADS),
        0,
        context_.hip_stream(), 
        static_cast<int>(input.size()),
        output.data<float>(),
        input.data<float>(),
        grad_output.data<float>(),
        grad_input->template mutable_data<float>());
  }
  return true;
}

/**
 * @brief Update slices of Y in-place with a batch of weighted X's.
 * Y[idx] = alpha[b] * X[b][i] + Y[idx]
 * i=0,...,N-1
 * b=0,...,B-1
 * idx=Indices[i]
 */
template <typename T_INDEX>
__global__ void AxpySliceKernel(
    const float* weight0,
    const int64_t N,
    const int64_t B,
    const int64_t slice_size,
    const float** alpha,
    const float** X,
    const T_INDEX* Indices,
    float* Y,
    const int64_t M) {
  // This implementation requires that the first weight is 1.0
  HIP_KERNEL_ASSERT(weight0[0] == 1.0);
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    T_INDEX idx = Indices[i];
    float* y_offset = Y + (idx * slice_size);
    for (int b = 0; b < B; b++) {
      float a = *alpha[b];
      const float* x_offset = X[b] + (i * slice_size);
      for (int j = threadIdx.x; j < slice_size; j += blockDim.x) {
        atomicAdd(&y_offset[j], a * x_offset[j]);
      }
    }
  }
}

template <>
bool ScatterWeightedSumOp<float, HIPContext>::RunOnDevice() {
  return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(2));
}

template <>
template <typename Index>
bool ScatterWeightedSumOp<float, HIPContext>::DoRunWithType() {
  CAFFE_ENFORCE_EQ(InputSize() % 2, 1);
  auto& X0 = Input(0);
  auto& weight0 = Input(1);
  auto& indices = Input(2);
  auto* output = Output(0);

  CAFFE_ENFORCE_EQ(&X0, output, "In place operation is required");
  CAFFE_ENFORCE_GT(X0.size(), 0);
  CAFFE_ENFORCE_GT(X0.ndim(), 0, "X0 has to be at least the vector");
  CAFFE_ENFORCE_EQ(weight0.size(), 1);

  int64_t M = X0.size();
  int64_t N = X0.dim(0);
  int64_t K = indices.size();
  int64_t block_size = M / N;

  float* data = output->template mutable_data<float>();

  // In order to have all device pointers of x_i (and weight_i similarly)
  // consecutively in device memory, copy pointers to a host vector and then
  // copy back into a device array.
  const int64_t B = (InputSize() - 3) / 2;
  x_data_host_.Resize(B);
  weights_host_.Resize(B);
  x_data_device_.Resize(B);
  weights_device_.Resize(B);

  const float** x_data_host = x_data_host_.mutable_data<const float*>();
  const float** weights_host = weights_host_.mutable_data<const float*>();
  const float** x_data_device = x_data_device_.mutable_data<const float*>();
  const float** weights_device = weights_device_.mutable_data<const float*>();

  for (int inp = 3; inp < InputSize(); inp += 2) {
    int idx = (inp - 3) / 2;
    x_data_host[idx] = static_cast<const float*>(Input(inp).raw_data());
    weights_host[idx] = static_cast<const float*>(Input(inp + 1).raw_data());
  }
  context_.Copy<const float*, CPUContext, HIPContext>(
      B, x_data_host, x_data_device);
  context_.Copy<const float*, CPUContext, HIPContext>(
      B, weights_host, weights_device);

 hipLaunchKernelGGL( AxpySliceKernel, 
      dim3(std::min<int64_t>(K, CAFFE_MAXIMUM_NUM_BLOCKS)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      weight0.template data<float>(),
      K,
      B,
      block_size,
      weights_device,
      x_data_device,
      indices.template data<Index>(),
      data,
      M);

  return true;
}

REGISTER_HIP_OPERATOR(
    ScatterWeightedSum,
    ScatterWeightedSumOp<float, HIPContext>);

namespace {

template <typename Index, typename T>
__global__ void scatter_assign_kernel(
    T* data,
    const Index* idxs,
    const T* slicesData,
    int64_t N,
    int64_t K,
    int64_t block_size) {
  for (int64_t i = blockIdx.x; i < K; i += gridDim.x) {
    Index idx = idxs[i];
    HIP_KERNEL_ASSERT(0 <= idx && idx < N);
    const T* src = slicesData + block_size * i;
    T* dest = data + block_size * idx;
    for (int64_t j = threadIdx.x; j < block_size; j += blockDim.x) {
      dest[j] = src[j];
    }
  }
}

} // namespace

template <>
template <typename Index, typename T>
void ScatterAssignOp<HIPContext>::DoScatterAssign(
    T* data,
    const Index* idxs,
    const T* slicesData,
    int64_t N,
    int64_t K,
    int64_t block_size) {
 hipLaunchKernelGGL( scatter_assign_kernel, 
      dim3(::min(K, static_cast<int64_t>(CAFFE_MAXIMUM_NUM_BLOCKS))),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), data, idxs, slicesData, static_cast<int64_t>(N), static_cast<int64_t>(K), static_cast<int64_t>(block_size));
}

REGISTER_HIP_OPERATOR(ScatterAssign, ScatterAssignOp<HIPContext>);

REGISTER_HIP_OPERATOR(Size, SizeOp<HIPContext>);

template <typename T>
__global__ void RangeKernel(const int n, T* Y, T offset, T step) {
  HIP_1D_KERNEL_LOOP(index, n) {
    Y[index] = index * step + offset;
  }
}

template <>
template <typename T>
bool RangeOp<HIPContext>::DoRunOnDevice(
    const T& start,
    const T& step,
    Tensor* output) {
  int N = output->size();
 hipLaunchKernelGGL( RangeKernel, 
      dim3(CAFFE_GET_BLOCKS(N)),
      dim3(CAFFE_HIP_NUM_THREADS),
      0,
      context_.hip_stream(), 
      static_cast<const int>(N), output->template mutable_data<T>(), start, step);
  return true;
}

REGISTER_HIP_OPERATOR(Range, RangeOp<HIPContext>);
} // namespace caffe2
