#pragma once

#include "ATen/ATen.h"
#include "ATen/Config.h"

#include <string>
#include <stdexcept>
#include <sstream>
#include <hipfft.h>
#include <hipfft.h>

namespace at { namespace native {

// This means that max dim is 3 + 2 = 5 with batch dimension and possible
// complex dimension
constexpr int max_rank = 3;

static inline std::string _cudaGetErrorEnum(hipfftResult error)
{
  switch (error)
  {
    case HIPFFT_SUCCESS:
      return "HIPFFT_SUCCESS";
    case HIPFFT_INVALID_PLAN:
      return "HIPFFT_INVALID_PLAN";
    case HIPFFT_ALLOC_FAILED:
      return "HIPFFT_ALLOC_FAILED";
    case HIPFFT_INVALID_TYPE:
      return "HIPFFT_INVALID_TYPE";
    case HIPFFT_INVALID_VALUE:
      return "HIPFFT_INVALID_VALUE";
    case HIPFFT_INTERNAL_ERROR:
      return "HIPFFT_INTERNAL_ERROR";
    case HIPFFT_EXEC_FAILED:
      return "HIPFFT_EXEC_FAILED";
    case HIPFFT_SETUP_FAILED:
      return "HIPFFT_SETUP_FAILED";
    case HIPFFT_INVALID_SIZE:
      return "HIPFFT_INVALID_SIZE";
    case HIPFFT_UNALIGNED_DATA:
      return "HIPFFT_UNALIGNED_DATA";
    case HIPFFT_INCOMPLETE_PARAMETER_LIST:
      return "HIPFFT_INCOMPLETE_PARAMETER_LIST";
    case HIPFFT_INVALID_DEVICE:
      return "HIPFFT_INVALID_DEVICE";
    case HIPFFT_PARSE_ERROR:
      return "HIPFFT_PARSE_ERROR";
    case HIPFFT_NO_WORKSPACE:
      return "HIPFFT_NO_WORKSPACE";
    case HIPFFT_NOT_IMPLEMENTED:
      return "HIPFFT_NOT_IMPLEMENTED";
#ifndef __HIP_PLATFORM_HCC__
    case HIPFFT_LICENSE_ERROR:
      return "HIPFFT_LICENSE_ERROR";
#endif
    case HIPFFT_NOT_SUPPORTED:
      return "HIPFFT_NOT_SUPPORTED";
    default:
      std::ostringstream ss;
      ss << "unknown error " << error;
      return ss.str();
  }
}

static inline void CUFFT_CHECK(hipfftResult error)
{
  if (error != HIPFFT_SUCCESS) {
    std::ostringstream ss;
    ss << "cuFFT error: " << _cudaGetErrorEnum(error);
    AT_ERROR(ss.str());
  }
}

}} // at::native
