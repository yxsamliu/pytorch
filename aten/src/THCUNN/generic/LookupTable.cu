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
assert(0);
}

#define THREADS 256
#define RUN(NORM, IDXTYPE) \
  calculate_norms_and_renorm<scalar_t, accreal, IDXTYPE, NORM> \
    <<<numel, THREADS/2, THREADS * sizeof(accreal), THCState_getCurrentStream(state)>>> \
    (weightsRaw, idxRaw, normType, maxNorm, THCTensor_(stride)(state, weight, 0))

void THNN_(LookupTable_renorm)(
           THCState *state,
           THCIndexTensor *idx,
           THCTensor *weight,
           accreal maxNorm,
           accreal normType){
assert(0);
}

#endif
