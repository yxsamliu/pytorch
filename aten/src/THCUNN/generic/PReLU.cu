#include "hip/hip_runtime.h"
#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/PReLU.cu"
#else

void THNN_(PReLU_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight)
{
  THCTensor_(resizeAs)(state, output, input);
  int64_t nOutputPlane = THCTensor_(numel)(state, weight);

  weight = THCTensor_(newContiguous)(state, weight);
  real *w = THCTensor_(data)(state, weight);

  if (nOutputPlane == 1)
  {
    THC_pointwiseApply2<real, real>(state, output, input, PReLUUpdateOutput<real>(w));
  }
  else
  {
    int ndim = THCTensor_(nDimensionLegacyAll)(state, input);
    input = THCTensor_(newContiguous)(state, input);

    int n = THCTensor_(nElement)(state, input);
    if (THTensor_sizeLegacyNoScalars(input, ndim > 1) != nOutputPlane)
      THError("Wrong number of input planes. Expected %d but got %d.", nOutputPlane, THTensor_sizeLegacyNoScalars(input, ndim > 1));

    int mapSize = 1;
    for (int d = 2; d < ndim; d++) {
      mapSize *= input->size(d);
    }
    int nElemsPerSample = nOutputPlane * mapSize;
   hipLaunchKernelGGL( preluForward<real>, dim3(GET_BLOCKS(n)), dim3(CUDA_NUM_THREADS), 0, THCState_getCurrentStream(state), 
      THCTensor_(data)(state, output),
      THCTensor_(data)(state, input),
      w,
      static_cast<int>(n), static_cast<int>(nElemsPerSample), static_cast<int>(mapSize)
    );
    THCudaCheck(hipGetLastError());
    THCTensor_(free)(state, input);
  }

  THCTensor_(free)(state, weight);
}

void THNN_(PReLU_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *weight)
{
  THCUNN_check_nElement(state, input, gradOutput);
  THCTensor_(resizeAs)(state, gradInput, input);
  int64_t nOutputPlane = THCTensor_(numel)(state, weight);

  weight = THCTensor_(newContiguous)(state, weight);
  real *w = THCTensor_(data)(state, weight);
  if (nOutputPlane == 1)
  {
    THC_pointwiseApply3<real, real, real>(state, gradInput, gradOutput, input, PReLUUpdateGradInput<real>(w));
  }
  else
  {
    int ndim = THCTensor_(nDimensionLegacyAll)(state, input);
    input = THCTensor_(newContiguous)(state, input);
    gradOutput = THCTensor_(newContiguous)(state, gradOutput);

    int n = THCTensor_(nElement)(state, input);
    if (THTensor_sizeLegacyNoScalars(input, ndim > 1) != nOutputPlane)
      THError("Wrong number of input planes. Expected %d but got %d.", nOutputPlane, THTensor_sizeLegacyNoScalars(input, ndim > 1));

    int mapSize = 1;
    for (int d = 2; d < ndim; d++) {
      mapSize *= input->size(d);
    }
    int nElemsPerSample = nOutputPlane * mapSize;
   hipLaunchKernelGGL( preluBackward<real>, dim3(GET_BLOCKS(n)), dim3(CUDA_NUM_THREADS), 0, THCState_getCurrentStream(state), 
      THCTensor_(data)(state, gradInput),
      THCTensor_(data)(state, input),
      w,
      THCTensor_(data)(state, gradOutput),
      static_cast<int>(n), static_cast<int>(nElemsPerSample), static_cast<int>(mapSize)
    );
    THCudaCheck(hipGetLastError());
    THCTensor_(free)(state, input);
    THCTensor_(free)(state, gradOutput);
  }
  THCTensor_(free)(state, weight);
}

void THNN_(PReLU_accGradParameters)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *weight,
           THCTensor *gradWeight,
           accreal scale_)
{
  real scale = ScalarConvert<accreal, real>::to(scale_);
  THCUNN_check_nElement(state, input, gradOutput);
  int64_t nOutputPlane = THCTensor_(numel)(state, weight);
  // use grad input for temporary storage, then call updateGradInput again

  if (nOutputPlane == 1)
  {
    THC_pointwiseApply3<real, real, real>(state, gradInput, input, gradOutput, PReLUAccGradParametersShared<real>());

    // introduces a sync point
    real sum = ScalarConvert<accreal, real>::to(THCTensor_(sumall)(state, gradInput));
    real w = THCTensor_(get1d)(state, gradWeight, 0);
    THCTensor_(set1d)(state, gradWeight, 0, w + sum * scale);

    // restore gradInput
    THNN_(PReLU_updateGradInput)(state, input, gradOutput, gradInput, weight);
  }
  else
  {
    int ndim = THCTensor_(nDimensionLegacyAll)(state, input);

    if (ndim == 1)
    {
      THC_pointwiseApply3<real, real, real>(state, gradWeight, input, gradOutput, PReLUAccGradParameters1to1<real>(scale));
    }
    else
    {
      THC_pointwiseApply3<real, real, real>(state, gradInput, input, gradOutput, PReLUAccGradParameters<real>(scale));
      THCTensor *gradWeightBuf = THCTensor_(new)(state);
      THCTensor_(resizeAs)(state, gradWeightBuf, gradWeight);

      if (ndim == 2)
      {
        THCTensor_(sum)(state, gradWeightBuf, gradInput, 0, 0);
        THCTensor_(cadd)(state, gradWeight, gradWeight, scale, gradWeightBuf);
      }
      else
      {
        THCTensor *sumbuf = THCTensor_(new)(state);
        THCTensor *buffer = THCTensor_(newContiguous)(state, gradInput);
        int64_t size3 = 1;
        for (int d = 2; d < ndim; d++) {
          size3 *= input->size(d);
        }
        THCTensor_(resize3d)(state, buffer, input->size(0), nOutputPlane, size3);
        THCTensor_(resize2d)(state, sumbuf, input->size(0), nOutputPlane);
        THCTensor_(sum)(state, sumbuf, buffer, 2, 0);
        THCTensor_(sum)(state, gradWeightBuf, sumbuf, 0, 0);
        THCTensor_(cadd)(state, gradWeight, gradWeight, scale, gradWeightBuf);
        THCTensor_(free)(state, buffer);
        THCTensor_(free)(state, sumbuf);
      }

      THCTensor_(free)(state, gradWeightBuf);

      // restore gradInput
      THNN_(PReLU_updateGradInput)(state, input, gradOutput, gradInput, weight);
    }
  }
}

#endif
