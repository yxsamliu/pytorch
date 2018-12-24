#pragma once

#include "ATen/Tensor.h"
#include "ATen/core/Half.h"

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace at {
template <>
inline __half* Tensor::data() const {
  return reinterpret_cast<__half*>(data<Half>());
}
} // namespace at
