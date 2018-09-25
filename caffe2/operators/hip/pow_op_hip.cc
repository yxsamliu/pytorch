#include "hip/hip_runtime.h"
#define CUB_STDERR
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#include "caffe2/core/hip/common_hip.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/pow_op.h"
#include "caffe2/utils/conversions.h"

namespace caffe2 {

// pow, log and other math functions are defined in
// HIP math library in header file math.h
#define HIP_POW(x, y) (pow(x, y))

template <int b_is_scalar, typename T1, typename T2, typename R>
__global__ void PowKernel(const T1* a, const T2* b, T2 e, R* out, int n) {
  HIP_1D_KERNEL_LOOP(i, n) {
    out[i] = HIP_POW(a[i], ((b == NULL) ? e : b[b_is_scalar ? 0 : i]));
  }
}
template <typename T1, typename T2, typename R>
__global__ void
PowBroadcastKernel(const T1* a, const T2* b, R* out, int pre, int n) {
  HIP_1D_KERNEL_LOOP(i, pre * n) {
    out[i] = HIP_POW(a[i], b[i % n]);
  }
}
template <typename T1, typename T2, typename R>
__global__ void PowBroadcast2Kernel(
    const T1* a,
    const T2* b,
    R* out,
    int pre,
    int n,
    int post) {
  HIP_1D_KERNEL_LOOP(i, pre * n * post) {
    out[i] = HIP_POW(a[i], b[(i / post) % n]);
  }
}

struct HipPowFunctor {
  template <bool b_is_scalar, typename T1, typename T2, typename R>
  inline void
  Run(size_t n, const T1* a, const T2* b, T2 e, R* out, HIPContext* context) {
   hipLaunchKernelGGL( PowKernel<b_is_scalar, T1, T2, R>
        , dim3(CAFFE_GET_BLOCKS(n)),
           dim3(CAFFE_HIP_NUM_THREADS),
           0,
           context->hip_stream(), a, b, e, out, static_cast<int>(n));
  }
  template <typename T1, typename T2, typename R>
  void RunWithBroadcast(
      const T1* a,
      const T2* b,
      R* out,
      size_t pre,
      size_t n,
      HIPContext* context) {
   hipLaunchKernelGGL( PowBroadcastKernel<T1, T2, R>
        , dim3(CAFFE_GET_BLOCKS(pre * n)),
           dim3(CAFFE_HIP_NUM_THREADS),
           0,
           context->hip_stream(), a, b, out, static_cast<int>(pre), static_cast<int>(n));
  }
  template <typename T1, typename T2, typename R>
  void RunWithBroadcast2(
      const T1* a,
      const T2* b,
      R* out,
      size_t pre,
      size_t n,
      size_t post,
      HIPContext* context) {
   hipLaunchKernelGGL( PowBroadcast2Kernel<T1, T2, R>
        , dim3(CAFFE_GET_BLOCKS(pre * n * post)),
           dim3(CAFFE_HIP_NUM_THREADS),
           0,
           context->hip_stream(), a, b, out, static_cast<int>(pre), static_cast<int>(n), static_cast<int>(post));
  }
};
REGISTER_HIP_OPERATOR(
    Pow,
    PowOp<
        TensorTypes<float> /*NumericTypes*/,
        HIPContext,
        HipPowFunctor,
        SameTypeAsInput>)

} // namespace caffe2
