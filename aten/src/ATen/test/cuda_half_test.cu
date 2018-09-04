#include "hip/hip_runtime.h"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/ATen.h"
#include "ATen/cuda/NumericLimits.cuh"
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"

#include <assert.h>

using namespace at;

__device__ void test(){
  
  // test half construction and implicit conversions in device
  ;
  ;
  // there is no float <=> __half implicit conversion
  ;

  __half a = __float2half(3.0f);
  __half b = __float2half(2.0f);
  __half c = a - Half(b);
  ;

  // asserting if the  functions used on 
  // half types give almost equivalent results when using
  //  functions on double.
  // The purpose of these asserts are to test the device side
  // half API for the common mathematical functions.
  // Note: When calling std math functions from device, don't
  // use the std namespace, but just "::" so that the function
  // gets resolved from nvcc math_functions.hpp

  float threshold = 0.00001;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  ;
  // note: can't use  namespace on isnan and isinf in device code
  #ifdef _MSC_VER
    // Windows requires this explicit conversion. The reason is unclear
    // related issue with clang: https://reviews.llvm.org/D37906
    ;
    ;
  #else
    ;
    ;
  #endif
}

__global__ void kernel(){
  test();
}

void launch_function(){
 hipLaunchKernelGGL( kernel, dim3(1),dim3()1, 0, 0, );
}

TEST_CASE( "half common math functions tests in device", "[cuda]" ) {
  launch_function();
  hipError_t err = hipDeviceSynchronize();
  REQUIRE(err == hipSuccess);
}

