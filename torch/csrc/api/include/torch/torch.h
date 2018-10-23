#pragma once

#include <torch/hip/hip_runtime.h>
#include <torch/jit.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <torch/serialize.h>
#include <torch/tensor.h>
#include <torch/utils.h>

#ifdef TORCH_API_INCLUDE_EXTENSION_H
#include <torch/extension.h>
#warning "Including torch/torch.h for C++ extensions is deprecated. Please include torch/extension.h"
#endif // defined(TORCH_API_INCLUDE_EXTENSION_H)
