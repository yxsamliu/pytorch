#include "hip/hip_runtime.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/AccumulateType.h"

#include <hiprand.h>
#include <hiprand_kernel.h>
#include <hiprand_kernel.h>
#include <utility>
#include <functional>


#include "ATen/native/Distributions.h"

#include <THC/THCGeneral.h>
#include <THC/THCTensorRandom.h>
#include <THC/THCGenerator.hpp>
#include <THC/THCApply.cuh>

#include <cstdint>
#include <limits>
#include <utility>

THCGenerator* THCRandom_getGenerator(THCState* state);

namespace {
std::pair<uint64_t, uint64_t> next_philox_seed(at::Generator* gen, uint64_t increment) {
  auto gen_ = THCRandom_getGenerator(at::globalContext().getTHCState());
  uint64_t offset = gen_->state.philox_seed_offset.fetch_add(increment);
  return std::make_pair(gen_->state.initial_seed, offset);
}

template <typename scalar_t>
void poisson_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& lambda,
    std::pair<uint64_t, uint64_t> seeds){
assert(0);
}

template <typename scalar_t>
void gamma_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& alpha,
    std::pair<uint64_t, uint64_t> seeds){
assert(0);
}

template <typename scalar_t>
void gamma_grad_cuda_kernel(
    at::Tensor& ret,
    const at::Tensor& self,
    const at::Tensor& output){
assert(0);
}

} // namespace

namespace at { namespace native {
Tensor _s_poisson_cuda(const Tensor& lambda, Generator* gen){
assert(0);
}

Tensor _s_gamma_cuda(const Tensor& alpha, Generator* gen) {
  Tensor ret = alpha.type().tensor(alpha.sizes());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(ret.type(), "gamma", [&] {
     gamma_cuda_kernel<scalar_t>(ret, alpha, next_philox_seed(gen, 10));
   });
  return ret;
}

Tensor _standard_gamma_grad_cuda(const Tensor& self, const Tensor& output) {
  Tensor ret = self.type().tensor(self.sizes());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.type(), "_standard_gamma_grad", [&] {
     gamma_grad_cuda_kernel<scalar_t>(ret, self, output);
   });
  return ret;
}

}} // namespace at::native
