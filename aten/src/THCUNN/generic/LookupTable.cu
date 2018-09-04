#include "hip/hip_runtime.h"
#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/LookupTable.cu"
#else

void THNN_(LookupTable_accGradParameters)(
           THCState *state,
           THCIndexTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCIndexTensor *count,
           THCIndexTensor *sortedIndices,
           THCIndexTensor *origIndices,
           bool scaleGradByFreq,
           int paddingValue,
           accreal scale_){
;
}

#define THREADS 256
#define RUN(NORM, IDXTYPE) \
 hipLaunchKernelGGL( calculate_norms_and_renorm<real, accreal, IDXTYPE, NORM> \
    , dim3(numel), dim3(THREADS/2), THREADS * sizeof(accreal), THCState_getCurrentStream(state),  \
    weightsRaw, idxRaw, normType, maxNorm, THCTensor_(stride)(state, weight, 0))

void THNN_(LookupTable_renorm)(
           THCState *state,
           THCIndexTensor *idx,
           THCTensor *weight,
           accreal maxNorm,
           accreal normType){
;
}

#endif
