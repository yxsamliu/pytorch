#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/LeakyReLU.cu"
#else

#include "../common.h"

void THNN_(LeakyReLU_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           accreal negval_,
           bool inplace)
{
  real negval = ScalarConvert<accreal, real>::to(negval_);

  THCUNN_assertSameGPU(state, 2, input, output);

  if (inplace)
  {
    THC_pointwiseApply1<real>(state, input, LeakyReLUUpdateOutputIP<real>(negval));
    THCTensor_(set)(state, output, input);
  }
  else
  {
    THCTensor_(resizeAs)(state, output, input);
    THC_pointwiseApply2<real, real>(state, output, input, LeakyReLUUpdateOutput<real>(negval));
  }

  THCudaCheck(hipGetLastError());
}

void THNN_(LeakyReLU_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           accreal negval_,
           bool inplace)
{
  real negval = ScalarConvert<accreal, real>::to(negval_);

  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, input, gradInput, gradOutput);

  if (inplace)
  {
    THC_pointwiseApply2<real, real>(state, gradOutput, input, LeakyReLUUpdateGradInputIP<real>(negval));
    THCTensor_(set)(state, gradInput, gradOutput);
  }
  else
  {
    THCTensor_(resizeAs)(state, gradInput, input);
    THC_pointwiseApply3<real, real, real>(state, gradInput, input, gradOutput, LeakyReLUUpdateGradInput<real>(negval));
  }

  THCudaCheck(hipGetLastError());
}

#endif
