#include "THCBlas.h"
#include "THCGeneral.h"
#include "TH/THHalf.h"

#include <algorithm>

float THCudaBlas_Sdot(THCState *state, int64_t n, float *x, int64_t incx, float *y, int64_t incy)
{
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    float result;
    rocblas_handle handle = THCState_getCurrentBlasHandle(state);
    rocblas_set_stream(handle, THCState_getCurrentStream(state));
    THCublasCheck(rocblas_sdot(handle, i_n, x, i_incx, y, i_incy, &result));
    return result;
  }

  THError("Cublas_Sdot only supports n, incx and incy "
          "up to signed integer limits: %d", INT_MAX);
  return 0;
}

double THCudaBlas_Ddot(THCState *state, int64_t n, double *x, int64_t incx, double *y, int64_t incy)
{
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    double result;
    rocblas_handle handle = THCState_getCurrentBlasHandle(state);
    rocblas_set_stream(handle, THCState_getCurrentStream(state));
    THCublasCheck(rocblas_ddot(handle, i_n, x, i_incx, y, i_incy, &result));
    return result;
  }

  THError("Cublas_Ddot only supports n, incx and incy "
          "up to signed integer limits: %d", INT_MAX);
  return 0;
}

at::Half THCudaBlas_Hdot(THCState *state, int64_t n, at::Half *x, int64_t incx, at::Half *y, int64_t incy)
{
#if CUDA_VERSION >= 8000
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    at::Half result;
    rocblas_handle handle = THCState_getCurrentBlasHandle(state);
    rocblas_set_stream(handle, THCState_getCurrentStream(state));
    THCublasCheck(rocblas_dotex(handle, n,
                              x, hipR16F, incx,
                              y, hipR16F, incy,
                              &result, hipR16F,
                              hipR32F));
    return result;
  }

  THError("Cublas_Hdot only supports n, incx and incy "
          "up to signed integer limits: %d", INT_MAX);
  return 0.0;
#else
  THError("Cublas_Hdot requires CUDA 8.0+");
  return 0.0;
#endif
}

/* Level 2 */

void adjustLdLevel2(int64_t m, int64_t n, int64_t *lda)
{
  // Note: leading dimensions generally are checked that they are > 0 and at least as big the result
  // requires (even if the value won't be used).
  // TODO: why does Level3 check trans but this doesn't?
  if (n <= 1)
    *lda = std::max<int64_t>(m, 1);
}

void THCudaBlas_Sgemv(THCState *state, char trans, int64_t m, int64_t n, float alpha, float *a, int64_t lda, float *x, int64_t incx, float beta, float *y, int64_t incy)
{
  adjustLdLevel2(m, n, &lda);

  rocblas_operation op;
  if (trans == 't') op = rocblas_operation_transpose;
  else if (trans == 'n') op = rocblas_operation_none;
  else if (trans == 'c') op = rocblas_operation_conjugate_transpose;
  else THError("Cublas_Sgemv parameter trans should be 't', 'n' or 'c'.");

  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    rocblas_handle handle = THCState_getCurrentBlasHandle(state);
    rocblas_set_stream(handle, THCState_getCurrentStream(state));
    THCublasCheck(rocblas_sgemv(handle, op, i_m, i_n, &alpha, a, i_lda, x, i_incx, &beta, y, i_incy));
    return;
  }
  THError("Cublas_Sgemv only supports m, n, lda, incx, incy"
          "in the range 0 < [val] <= %d", INT_MAX);
}

void THCudaBlas_Dgemv(THCState *state, char trans, int64_t m, int64_t n, double alpha, double *a, int64_t lda, double *x, int64_t incx, double beta, double *y, int64_t incy)
{
  adjustLdLevel2(m, n, &lda);

  rocblas_operation op;
  if (trans == 't') op = rocblas_operation_transpose;
  else if (trans == 'n') op = rocblas_operation_none;
  else if (trans == 'c') op = rocblas_operation_conjugate_transpose;
  else THError("Cublas_Sgemv parameter trans should be 't', 'n' or 'c'.");

  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    rocblas_handle handle = THCState_getCurrentBlasHandle(state);
    rocblas_set_stream(handle, THCState_getCurrentStream(state));
    THCublasCheck(rocblas_dgemv(handle, op, i_m, i_n, &alpha, a, i_lda, x, i_incx, &beta, y, i_incy));
    return;
  }
  THError("Cublas_Dgemv only supports m, n, lda, incx, incy"
          "in the range 0 < [val] <= %d", INT_MAX);
}

void THCudaBlas_Sger(THCState *state, int64_t m, int64_t n, float alpha, float *x, int64_t incx, float *y, int64_t incy, float *a, int64_t lda)
{
  adjustLdLevel2(m, n, &lda);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;

      rocblas_handle handle = THCState_getCurrentBlasHandle(state);
      rocblas_set_stream(handle, THCState_getCurrentStream(state));
      THCublasCheck(rocblas_sger(handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
      return;
    }
  THError("Cublas_Sger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}

void THCudaBlas_Dger(THCState *state, int64_t m, int64_t n, double alpha, double *x, int64_t incx, double *y, int64_t incy, double *a, int64_t lda)
{
  adjustLdLevel2(m, n, &lda);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;

      rocblas_handle handle = THCState_getCurrentBlasHandle(state);
      rocblas_set_stream(handle, THCState_getCurrentStream(state));
      THCublasCheck(rocblas_dger(handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
      return;
    }
  THError("Cublas_Dger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}


rocblas_operation convertTransToCublasOperation(char trans) {
  if (trans == 't') return rocblas_operation_transpose;
  else if (trans == 'n') return rocblas_operation_none;
  else if (trans == 'c') return rocblas_operation_conjugate_transpose;
  else {
    THError("trans must be one of: t, n, c");
    return rocblas_operation_transpose;
  }
}

void adjustLdLevel3(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t *lda, int64_t *ldb, int64_t *ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  // Note: leading dimensions generally are checked that they are > 0 and at least as big the result
  // requires (even if the value won't be used).
  if(n <= 1)
    *ldc = std::max<int64_t>(m, 1);

  if(transa_)
  {
    if(m <= 1)
      *lda = std::max<int64_t>(k, 1);
  }
  else
  {
    if(k <= 1)
      *lda = std::max<int64_t>(m, 1);
  }

  if(transb_)
  {
    if(k <= 1)
      *ldb = std::max<int64_t>(n, 1);
  }
  else
  {
    if(n <= 1)
      *ldb = std::max<int64_t>(k, 1);
  }

}

/* Level 3 */
void THCudaBlas_Sgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, float *a, int64_t lda, float *b, int64_t ldb, float beta, float *c, int64_t ldc)
{
  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  rocblas_operation opa = convertTransToCublasOperation(transa);
  rocblas_operation opb = convertTransToCublasOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    rocblas_handle handle = THCState_getCurrentBlasHandle(state);
    rocblas_set_stream(handle, THCState_getCurrentStream(state));
    THCublasCheck(rocblas_sgemm(handle, opa, opb, i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb, &beta, c, i_ldc));
    return;
  }
  THError("Cublas_Sgemm only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}

// In CUDA 8.0, definition of data types for sgemmex changed
#if CUDA_VERSION < 8000
#  define hipR16F rocblas_precision_half
#endif

void THCudaBlas_Hgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, at::Half alpha, at::Half *a, int64_t lda, at::Half *b, int64_t ldb, at::Half beta, at::Half *c, int64_t ldc)
{
  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  rocblas_operation opa = convertTransToCublasOperation(transa);
  rocblas_operation opb = convertTransToCublasOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_k = (int)k;
      int i_lda = (int)lda;
      int i_ldb = (int)ldb;
      int i_ldc = (int)ldc;

      rocblas_handle handle = THCState_getCurrentBlasHandle(state);
      rocblas_set_stream(handle, THCState_getCurrentStream(state));

#ifdef __HIP_PLATFORM_HCC__
#if __hcc_workweek__ < 18451
      THCublasCheck(rocblas_hgemm(handle, opa, opb, i_m, i_n, i_k,
                    reinterpret_cast<rocblas_half*>(&alpha), reinterpret_cast<rocblas_half*>(a), i_lda,
                    reinterpret_cast<rocblas_half*>(b), i_ldb, reinterpret_cast<rocblas_half*>(&beta),
                    reinterpret_cast<rocblas_half*>(c), i_ldc));
#else
     float fAlpha = alpha;
     float fBeta = beta;
     THCublasCheck(rocblas_gemm_ex(handle, opa, opb, i_m, i_n, i_k,
                   &fAlpha, a, rocblas_datatype_f16_r, i_lda, b, rocblas_datatype_f16_r,
                   i_ldb, &fBeta, c, rocblas_datatype_f16_r, i_ldc, c, rocblas_datatype_f16_r,
                   i_ldc, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0, NULL, NULL));
#endif
#else

      // Simulated Hgemm
      float fAlpha = alpha;
      float fBeta = beta;

#if CUDA_VERSION < 9000
      THCublasCheck(rocblas_status_internal_error);
#else
      hipDeviceProp_t* prop = THCState_getCurrentDeviceProperties(state);
      if (prop->major >= 5){
        THCublasCheck(rocblas_set_math_mode(handle, CUBLAS_TENSOR_OP_MATH));
	THCublasCheck(rocblas_gemmex(handle, opa, opb,
                                   i_m, i_n, i_k, &fAlpha,
                                   a, hipR16F, i_lda, b, hipR16F,
                                   i_ldb, &fBeta, c, hipR16F, i_ldc,
                                   hipR32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
	THCublasCheck(rocblas_set_math_mode(handle, CUBLAS_DEFAULT_MATH));
      }else{
        THCublasCheck(rocblas_status_internal_error);
      }
#endif
#endif
      return;
    }
  THError("Cublas_Hgemm only supports m, n, k, lda, ldb, ldc"
          "with th bound [val] <= %d", INT_MAX);
}

void THCudaBlas_Dgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, double alpha, double *a, int64_t lda, double *b, int64_t ldb, double beta, double *c, int64_t ldc)
{
  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  rocblas_operation opa = convertTransToCublasOperation(transa);
  rocblas_operation opb = convertTransToCublasOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    rocblas_handle handle = THCState_getCurrentBlasHandle(state);
    rocblas_set_stream(handle, THCState_getCurrentStream(state));
    THCublasCheck(rocblas_dgemm(handle, opa, opb, i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb, &beta, c, i_ldc));
    return;
  }
  THError("Cublas_Dgemm only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}

#if CUDA_VERSION >= 9010  || (defined __HIP_PLATFORM_HCC__ && __hcc_workweek__ > 18451)
void THCudaBlas_HgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                             at::Half alpha, const at::Half *a, int64_t lda, int64_t strideA, const at::Half *b, int64_t ldb, int64_t strideB,
                             at::Half beta, at::Half *c, int64_t ldc, int64_t strideC, int64_t batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )

  {
    THError("Cublas_SgemmStridedBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  rocblas_operation opa = convertTransToCublasOperation(transa);
  rocblas_operation opb = convertTransToCublasOperation(transb);

  rocblas_handle handle = THCState_getCurrentBlasHandle(state);
  rocblas_set_stream(handle, THCState_getCurrentStream(state));
  float fAlpha = alpha;
  float fBeta = beta;
#ifdef __HIP_PLATFORM_HCC__
  THCublasCheck(rocblas_gemm_strided_batched_ex(handle, opa, opb, (int)m, (int)n, (int)k,
                                   (void*)&fAlpha, a, rocblas_datatype_f16_r, (int)lda, strideA,
                                   b, rocblas_datatype_f16_r, (int)ldb, strideB,
                                   (void*)&fBeta, c, rocblas_datatype_f16_r, (int)ldc, strideC,
                                   c, rocblas_datatype_f16_r, (int)ldc, strideC,
                                   (int) batchCount, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
                                   0, 0, NULL, NULL));
#else
  THCublasCheck(rocblas_set_math_mode(handle, CUBLAS_TENSOR_OP_MATH));
  THCublasCheck(cublasGemmStridedBatchedEx(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   (void*)&fAlpha, a, hipR16F, (int)lda, strideA,
                                   b, hipR16F, (int)ldb, strideB,
                                   (void*)&fBeta, c, hipR16F, (int)ldc, strideC,
                                   (int)batchCount, hipR32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  THCublasCheck(rocblas_set_math_mode(handle, CUBLAS_DEFAULT_MATH));
#endif
}
#endif

void THCudaBlas_SgemmBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                             float alpha, const float *a[], int64_t lda, const float *b[], int64_t ldb,
                             float beta, float *c[], int64_t ldc, int64_t batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_SgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

#ifdef __HIP_PLATFORM_HCC__

  const int64_t stridea = (transa == 'N' || transa == 'n') ? lda*k : lda*n;
  const int64_t strideb = (transb == 'N' || transb == 'n') ? ldb*n : ldb*k;
  const int64_t stridec = ldc*n;

  THCudaBlas_SgemmStridedBatched(state, transa, transb, m, n, k, alpha, *a, lda, stridea, *b, ldb, strideb, beta, *c, ldc, stridec, batchCount);

#else

  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  rocblas_operation opa = convertTransToCublasOperation(transa);
  rocblas_operation opb = convertTransToCublasOperation(transb);

  rocblas_handle handle = THCState_getCurrentBlasHandle(state);
  rocblas_set_stream(handle, THCState_getCurrentStream(state));
  THCublasCheck(rocblas_sgemm_batched(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   &alpha, a, (int)lda, b, (int)ldb, &beta, c, (int)ldc,
                                   (int)batchCount));
#endif
}

#if CUDA_VERSION >= 8000 || defined __HIP_PLATFORM_HCC__
void THCudaBlas_SgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                             float alpha, const float *a, int64_t lda, int64_t strideA, const float *b, int64_t ldb, int64_t strideB,
                             float beta, float *c, int64_t ldc, int64_t strideC, int64_t batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )

  {
    THError("Cublas_SgemmStridedBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  rocblas_operation opa = convertTransToCublasOperation(transa);
  rocblas_operation opb = convertTransToCublasOperation(transb);

  rocblas_handle handle = THCState_getCurrentBlasHandle(state);
  rocblas_set_stream(handle, THCState_getCurrentStream(state));
  THCublasCheck(rocblas_sgemm_strided_batched(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   &alpha, a, (int)lda, strideA, b, (int)ldb, strideB, &beta, c, (int)ldc, strideC,
                                   (int)batchCount));
}
#endif

void THCudaBlas_DgemmBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                             double alpha, const double *a[], int64_t lda, const double *b[], int64_t ldb,
                             double beta, double *c[], int64_t ldc, int64_t batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_DgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

#ifdef __HIP_PLATFORM_HCC__

  const int64_t stridea = (transa == 'N' || transa == 'n') ? lda*k : lda*n;
  const int64_t strideb = (transb == 'N' || transb == 'n') ? ldb*n : ldb*k;
  const int64_t stridec = ldc*n;

  THCudaBlas_DgemmStridedBatched(state, transa, transb, m, n, k, alpha, *a, lda, stridea, *b, ldb, strideb, beta, *c, ldc, stridec, batchCount);

#else

  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  rocblas_operation opa = convertTransToCublasOperation(transa);
  rocblas_operation opb = convertTransToCublasOperation(transb);

  rocblas_handle handle = THCState_getCurrentBlasHandle(state);
  rocblas_set_stream(handle, THCState_getCurrentStream(state));
  THCublasCheck(rocblas_dgemm_batched(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   &alpha, a, (int)lda, b, (int)ldb, &beta, c, (int)ldc,
                                   (int)batchCount));
#endif
}

#if CUDA_VERSION >= 8000 || defined __HIP_PLATFORM_HCC__
void THCudaBlas_DgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                             double alpha, const double *a, int64_t lda, int64_t strideA, const double *b, int64_t ldb, int64_t strideB,
                             double beta, double *c, int64_t ldc, int64_t strideC, int64_t batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_DgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  rocblas_operation opa = convertTransToCublasOperation(transa);
  rocblas_operation opb = convertTransToCublasOperation(transb);

  rocblas_handle handle = THCState_getCurrentBlasHandle(state);
  rocblas_set_stream(handle, THCState_getCurrentStream(state));
  THCublasCheck(rocblas_dgemm_strided_batched(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   &alpha, a, (int)lda, strideA, b, (int)ldb, strideB, &beta, c, (int)ldc, strideC,
                                   (int)batchCount));
}
#endif

/* Inverse */
void THCudaBlas_Sgetrf(THCState *state, int n, float **a, int lda, int *pivot, int *info, int batchSize) {
  if( (n >= INT_MAX) || (lda >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Sgetrf only supports n, lda, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  rocblas_handle handle = THCState_getCurrentBlasHandle(state);
  rocblas_set_stream(handle, THCState_getCurrentStream(state));
  THCublasCheck(rocblas_status_internal_error);
}

void THCudaBlas_Dgetrf(THCState *state, int n, double **a, int lda, int *pivot, int *info, int batchSize) {
  if( (n >= INT_MAX) || (lda >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetrf only supports n, lda, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  rocblas_handle handle = THCState_getCurrentBlasHandle(state);
  rocblas_set_stream(handle, THCState_getCurrentStream(state));
  THCublasCheck(rocblas_status_internal_error);
}

void THCudaBlas_Sgetrs(THCState *state, char transa, int n, int nrhs, const float **a, int lda, int *pivot, float **b, int ldb, int *info, int batchSize)
{
  if( (n >= INT_MAX) || (nrhs >= INT_MAX) || (lda >= INT_MAX) || (ldb >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetrs only supports n, nrhs, lda, ldb, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }

  // no need to adjust leading dimensions, since matrices are square
  rocblas_operation opa = convertTransToCublasOperation(transa);

  rocblas_handle handle = THCState_getCurrentBlasHandle(state);
  rocblas_set_stream(handle, THCState_getCurrentStream(state));
  THCublasCheck(rocblas_status_internal_error);
}


void THCudaBlas_Dgetrs(THCState *state, char transa, int n, int nrhs, const double **a, int lda, int *pivot, double **b, int ldb, int *info, int batchSize)
{
  if( (n >= INT_MAX) || (nrhs >= INT_MAX) || (lda >= INT_MAX) || (ldb >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetrs only supports n, nrhs, lda, ldb, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }

  // no need to adjust leading dimensions, since matrices are square
  rocblas_operation opa = convertTransToCublasOperation(transa);

  rocblas_handle handle = THCState_getCurrentBlasHandle(state);
  rocblas_set_stream(handle, THCState_getCurrentStream(state));
  THCublasCheck(rocblas_status_internal_error);
}

void THCudaBlas_Sgetri(THCState *state, int n, const float **a, int lda, int *pivot, float **c, int ldc, int *info, int batchSize) {

  if( (n >= INT_MAX) || (lda >= INT_MAX)|| (ldc >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Sgetri only supports n, lda, ldc, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  rocblas_handle handle = THCState_getCurrentBlasHandle(state);
  rocblas_set_stream(handle, THCState_getCurrentStream(state));
  THCublasCheck(rocblas_status_internal_error);
}

void THCudaBlas_Dgetri(THCState *state, int n, const double **a, int lda, int *pivot, double **c, int ldc, int *info, int batchSize) {

  if( (n >= INT_MAX) || (lda >= INT_MAX)|| (ldc >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetri only supports n, lda, ldc, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  rocblas_handle handle = THCState_getCurrentBlasHandle(state);
  rocblas_set_stream(handle, THCState_getCurrentStream(state));
  THCublasCheck(rocblas_status_internal_error);
}
