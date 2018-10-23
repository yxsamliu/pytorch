#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SparseLinear.cu"
#else

static bool THNN_(checkInput)(THCTensor* t)
{
  return !t->is_empty() && t->dim() == 2 && t->size(1) == 3;
}

static bool THNN_(checkSize2D)(THCTensor* t, int64_t size0, int64_t size1)
{
  return !t->is_empty() && t->dim() == 2 && t->size(0) == size0 && t->size(1) == size1;
}

static bool THNN_(checkSize1D)(THCTensor* t, int64_t size0)
{
  return !t->is_empty() && THTensor_nDimensionLegacyNoScalars(t) == 1 && THTensor_sizeLegacyNoScalars(t, 0) == size0;
}

static inline void THNN_(copyCudaFloatingType)(THCState *state, THCudaIntTensor *buf, THCTensor *t) {
  #ifdef THC_REAL_IS_FLOAT
  THCudaIntTensor_copyCudaFloat(state, buf, t);
  #elif defined(THC_REAL_IS_DOUBLE)
  THCudaIntTensor_copyCudaDouble(state, buf, t);
  #elif defined(THC_REAL_IS_HALF)
  THCudaIntTensor_copyCudaHalf(state, buf, t);
  #endif
}

void THNN_(SparseLinear_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias){
assert(0);
}

void THNN_(SparseLinear_accGradParameters)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *weight,
           THCTensor *bias,
           accreal weightDecay,
           accreal scale){
assert(0);
}

void THNN_(SparseLinear_legacyUpdateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias) {
  THError("CUDA does not support legacy input format, please use a table of nnz x 2 vectors");
}
void THNN_(SparseLinear_legacyAccGradParameters)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *weight,
           THCTensor *bias,
           accreal weightDecay,
           accreal scale) {
  THError("CUDA does not support legacy input format, please use a table of nnz x 2 vectors");
}

// Dense updates are pretty fast on the GPU
void THNN_(SparseLinear_zeroGradParameters)(
           THCState *state,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *lastInput) {
  THCTensor_(zero)(state, gradWeight);
  THCTensor_(zero)(state, gradBias);
}

void THNN_(SparseLinear_updateParameters)(
           THCState *state,
           THCTensor *weight,
           THCTensor *bias,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *lastInput,
           accreal learningRate) {
  THCTensor_(cadd)(state, weight, weight, -learningRate, gradWeight);
  THCTensor_(cadd)(state, bias, bias, -learningRate, gradBias);
}

#endif
