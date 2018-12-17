#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_ROCM_FUSER

#include "ATen/ATen.h"
#include "torch/csrc/WindowsTorchApiMacro.h"
#include "torch/csrc/jit/fuser/fused_kernel.h"

#include "nvrtc.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime.h"

#include <cstdint>
#include <vector>
#include <string>

namespace torch { namespace jit { namespace fuser { namespace cuda {

// A class holding metadata for an actual CUDA function.
// Note: CUDA functions are per device.
struct TORCH_API FusedKernelCUDA : public ::torch::jit::fuser::FusedKernel {
  FusedKernelCUDA(
      int16_t device,
      std::string name,
      std::string code,
      std::vector<TensorDesc> input_desc,
      std::vector<TensorDesc> output_desc,
      std::vector<PartitionDesc> chunk_desc,
      std::vector<PartitionDesc> concat_desc,
      bool has_random);

  ~FusedKernelCUDA() override {
    hipModuleUnload(module_);
  }

  void launch_raw(const uint32_t numel, std::vector<void*>& arguments)
      const override;

  at::Backend backend() const override {
    return at::Backend::CUDA;
  }

private:
  static constexpr auto kBlockSize = 128;

  // Note: per device to store device properties and compute launch heuristics
  //  Acquiring these values at launch time would be too slow
  int16_t device_;
  int maxBlocks_;
  hipDeviceProp_t* prop_;
  std::vector<char> ptx_;
  hipModule_t module_;
  hipFunction_t function_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

#endif // USE_ROCM_FUSER
