#include "hip/hip_runtime.h"
#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialAdaptiveAveragePooling.cu"
#else

#include "../common.h"

// 4d tensor B x D x H x W

void THNN_(SpatialAdaptiveAveragePooling_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           int osizeW,
           int osizeH)
{
  THCUNN_assertSameGPU(state, 2, input, output);

  scalar_t *output_data;
  scalar_t *input_data;

  THCUNN_argCheck(state, !input->is_empty() && (input->dim() == 3 || input->dim() == 4), 2, input,
                  "non-empty 3D or 4D (batch mode) tensor expected for input, but got: %s");

  if (input->dim() == 3) {
    int64_t sizeD  = input->size(0);
    int64_t isizeH = input->size(1);
    int64_t isizeW = input->size(2);

    int64_t istrideD = input->stride(0);
    int64_t istrideH = input->stride(1);
    int64_t istrideW = input->stride(2);

    input_data = THCTensor_(data)(state, input);

    THCTensor_(resize3d)(state, output, sizeD, osizeH, osizeW);

    output_data = THCTensor_(data)(state, output);

    // cuda blocks & threads:
    int blocksH = max((int)(16L / sizeD), 1);
    dim3 blocks(sizeD, blocksH);
    dim3 threads(32, 8);

    // run averagepool kernel
   hipLaunchKernelGGL( adaptiveaveragepool<scalar_t> , dim3(blocks), dim3(threads), 0, THCState_getCurrentStream(state),  input_data, output_data,
                                   static_cast<int>(isizeH), static_cast<int>(isizeW), static_cast<int>(osizeH), static_cast<int>(osizeW),
                                   static_cast<int64_t>(istrideD), static_cast<int64_t>(istrideH), static_cast<int64_t>(istrideW));
    THCudaCheck(hipGetLastError());

  } else {
    input = THCTensor_(newContiguous)(state, input);
    int64_t sizeB  = input->size(0);
    int64_t sizeD  = input->size(1);
    int64_t isizeH = input->size(2);
    int64_t isizeW = input->size(3);

    int64_t istrideD = input->stride(1);
    int64_t istrideH = input->stride(2);
    int64_t istrideW = input->stride(3);

    input_data = THCTensor_(data)(state, input);

    THCTensor_(resize4d)(state, output, sizeB, sizeD, osizeH, osizeW);

    output_data = THCTensor_(data)(state, output);

    // cuda blocks & threads:
    int blocksH = max((int)(16L / sizeD), 1);
    dim3 blocks(sizeB * sizeD, blocksH);
    dim3 threads(32, 8);

    // run averagepool kernel
   hipLaunchKernelGGL( adaptiveaveragepool<scalar_t> , dim3(blocks), dim3(threads), 0, THCState_getCurrentStream(state),  input_data, output_data,
                                   static_cast<int>(isizeH), static_cast<int>(isizeW), static_cast<int>(osizeH), static_cast<int>(osizeW),
                                   static_cast<int64_t>(istrideD), static_cast<int64_t>(istrideH), static_cast<int64_t>(istrideW));
    THCudaCheck(hipGetLastError());
    // clean
    THCTensor_(free)(state, input);
  }
}

void THNN_(SpatialAdaptiveAveragePooling_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput)
{
  bool atomic = true; // suboptimal, but without atomic it doesn't pass the tests

  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);

  scalar_t *gradInput_data;
  scalar_t *gradOutput_data;

  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  if (input->dim() == 3) {
    int64_t sizeD  = input->size(0);
    int64_t isizeH = input->size(1);
    int64_t isizeW = input->size(2);

    int64_t osizeH = gradOutput->size(1);
    int64_t osizeW = gradOutput->size(2);

    //bool atomic = (isizeW%osizeW != 0) || (isizeH%osizeH != 0);

    THCTensor_(resizeAs)(state, gradInput, input);
    THCTensor_(zero)(state, gradInput);

    gradOutput_data = THCTensor_(data)(state, gradOutput);
    gradInput_data = THCTensor_(data)(state, gradInput);

    // cuda blocks & threads:
    int blocksH = max((int)(16L / sizeD), 1);
    dim3 blocks(sizeD, blocksH);
    dim3 threads(32, 8);

    if(atomic)
    {
      // run updateGradInput kernel, accumulate gradients atomically
     hipLaunchKernelGGL( atomicadaptiveaveragegradinput<scalar_t> , dim3(blocks), dim3(threads), 0, THCState_getCurrentStream(state),  gradInput_data, gradOutput_data,
                                          static_cast<int>(isizeH), static_cast<int>(isizeW), static_cast<int>(osizeH), static_cast<int>(osizeW));
    }
    else
    {
      // run updateGradInput kernel
     hipLaunchKernelGGL( adaptiveaveragegradinput<scalar_t> , dim3(blocks), dim3(threads), 0, THCState_getCurrentStream(state),  gradInput_data, gradOutput_data,
                                          static_cast<int>(isizeH), static_cast<int>(isizeW), static_cast<int>(osizeH), static_cast<int>(osizeW));
    }
    THCudaCheck(hipGetLastError());
  } else {
    int64_t sizeB  = input->size(0);
    int64_t sizeD  = input->size(1);
    int64_t isizeH = input->size(2);
    int64_t isizeW = input->size(3);

    int64_t osizeH = gradOutput->size(2);
    int64_t osizeW = gradOutput->size(3);

    //bool atomic = //(isizeW%osizeW != 0) || (isizeH%osizeH != 0);

    THCTensor_(resizeAs)(state, gradInput, input);
    THCTensor_(zero)(state, gradInput);

    gradOutput_data = THCTensor_(data)(state, gradOutput);
    gradInput_data = THCTensor_(data)(state, gradInput);

    // cuda blocks & threads:
    int blocksH = max((int)(16L / sizeD), 1);
    dim3 blocks(sizeB * sizeD, blocksH);
    dim3 threads(32, 8);

    if(atomic)
    {
      // run updateGradInput kernel, accumulate gradients atomically
     hipLaunchKernelGGL( atomicadaptiveaveragegradinput<scalar_t> , dim3(blocks), dim3(threads), 0, THCState_getCurrentStream(state),  gradInput_data, gradOutput_data,
                                          static_cast<int>(isizeH), static_cast<int>(isizeW), static_cast<int>(osizeH), static_cast<int>(osizeW));
    }
    else
    {
      // run updateGradInput kernel, accumulate gradients atomically
     hipLaunchKernelGGL( adaptiveaveragegradinput<scalar_t> , dim3(blocks), dim3(threads), 0, THCState_getCurrentStream(state),  gradInput_data, gradOutput_data,
                                          static_cast<int>(isizeH), static_cast<int>(isizeW), static_cast<int>(osizeH), static_cast<int>(osizeW));
    }
    THCudaCheck(hipGetLastError());
  }

  // clean
  THCTensor_(free)(state,gradOutput);

}

#endif
