#pragma once

#include <sstream>
#include <stdexcept>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

#include <ATen/ATen.h>
#include <THC/THCStream.h>

#include <c10d/CUDAUtils.hpp>

// TODO: Use AT_CHECK or similar here
#define C10D_CUDA_CHECK(condition)        \
  do {                                    \
    hipError_t error = (condition);      \
    if (error != hipSuccess) {           \
      std::stringstream ss;               \
      ss << "Error at: ";                 \
      ss << __FILE__;                     \
      ss << ":";                          \
      ss << __LINE__;                     \
      ss << ": ";                         \
      ss << hipGetErrorString(error);    \
      throw std::runtime_error(ss.str()); \
    }                                     \
  } while (0)
