#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <ATen/native/sparse/cuda/SparseCUDABlas.cuh>

#include <TH/THGeneral.h>

#include <hipsparse.h>

namespace at { namespace native { namespace sparse { namespace cuda {


std::string cusparseGetErrorString(hipsparseStatus_t status) {
  switch(status)
  {
    case HIPSPARSE_STATUS_SUCCESS:
      return "success";

    case HIPSPARSE_STATUS_NOT_INITIALIZED:
      return "library not initialized";

    case HIPSPARSE_STATUS_ALLOC_FAILED:
      return "resource allocation failed";

    case HIPSPARSE_STATUS_INVALID_VALUE:
      return "an invalid numeric value was used as an argument";

    case HIPSPARSE_STATUS_ARCH_MISMATCH:
      return "an absent device architectural feature is required";

    case HIPSPARSE_STATUS_MAPPING_ERROR:
      return "an access to GPU memory space failed";

    case HIPSPARSE_STATUS_EXECUTION_FAILED:
      return "the GPU program failed to execute";

    case HIPSPARSE_STATUS_INTERNAL_ERROR:
      return "an internal operation failed";

    case HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "the matrix type is not supported by this function";

    case HIPSPARSE_STATUS_ZERO_PIVOT:
      return "an entry of the matrix is either structural zero or numerical zero (singular block)";

    default:
      {
        std::ostringstream oss;
        oss << "unknown error " << static_cast<int64_t>(status);
        return oss.str();
      }
  }
}

inline void CUSPARSE_CHECK(hipsparseStatus_t status)
{
  if (status != HIPSPARSE_STATUS_SUCCESS) {
    AT_ERROR("cusparse runtime error: ", cusparseGetErrorString(status));
  }
}

inline hipsparseHandle_t setCUDASparseStream() {
  hipsparseHandle_t handle = at::cuda::getCurrentCUDASparseHandle();
  hipsparseSetStream(handle, at::cuda::getCurrentCUDAStream());
  return handle;
}

void Xcoo2csr(const int *coorowind, int64_t nnz, int64_t m, int *csrrowptr) {
  AT_CHECK((m <= INT_MAX) && (nnz <= INT_MAX),
    "hipsparseXcoo2csr only supports m, nnz with the bound [val] <= ",
    INT_MAX);
  auto handle = setCUDASparseStream();
  CUSPARSE_CHECK(hipsparseXcoo2csr(handle, coorowind, nnz, m, csrrowptr,
    TH_INDEX_BASE ? HIPSPARSE_INDEX_BASE_ONE : HIPSPARSE_INDEX_BASE_ZERO
  ));
}

hipsparseOperation_t convertTransToCusparseOperation(char trans) {
  if (trans == 't') return HIPSPARSE_OPERATION_TRANSPOSE;
  else if (trans == 'n') return HIPSPARSE_OPERATION_NON_TRANSPOSE;
  else if (trans == 'c') return HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else {
    AT_ERROR("trans must be one of: t, n, c");
  }
}

void adjustLd(char transb, int64_t m, int64_t n, int64_t k, int64_t *ldb, int64_t *ldc)
{
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    *ldc = m;

  if(transb_)
  {
    if(k == 1)
      *ldb = n;
  }
  else
  {
    if(n == 1)
      *ldb = k;
  }
}

/* Level 3 */
void Scsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, float alpha, float *csrvala, int *csrrowptra, int *csrcolinda, float *b, int64_t ldb, float beta, float *c, int64_t ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  hipsparseOperation_t opa = convertTransToCusparseOperation(transa);
  hipsparseOperation_t opb = convertTransToCusparseOperation(transb);

  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "hipsparseScsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;

  auto handle = setCUDASparseStream();
  hipsparseMatDescr_t desc;
  hipsparseCreateMatDescr(&desc);
#if TH_INDEX_BASE == 1
  hipsparseSetMatIndexBase(&desc, HIPSPARSE_INDEX_BASE_ONE);
#endif
  CUSPARSE_CHECK(hipsparseScsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, &alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, &beta, c, i_ldc));
}

void Dcsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, double alpha, double *csrvala, int *csrrowptra, int *csrcolinda, double *b, int64_t ldb, double beta, double *c, int64_t ldc)
{
  adjustLd(transb, m, n, k, &ldb, &ldc);
  hipsparseOperation_t opa = convertTransToCusparseOperation(transa);
  hipsparseOperation_t opb = convertTransToCusparseOperation(transb);

  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "hipsparseDcsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;

  auto handle = setCUDASparseStream();
  hipsparseMatDescr_t desc;
  hipsparseCreateMatDescr(&desc);
#if TH_INDEX_BASE == 1
  hipsparseSetMatIndexBase(&desc, HIPSPARSE_INDEX_BASE_ONE);
#endif
  CUSPARSE_CHECK(hipsparseDcsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, &alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, &beta, c, i_ldc));
  // TODO: I think this leaks the matrix descriptor.  Proper fix is to create
  // real descriptor classes
}

/* format conversion */
void CreateIdentityPermutation(int64_t nnz, int *P) {
  AT_CHECK((nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_nnz = (int)nnz;

  auto handle = setCUDASparseStream();
  hipsparseCreateIdentityPermutation(handle, i_nnz, P);
}

void Xcsrsort_bufferSizeExt(int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, const int *csrColInd, size_t *pBufferSizeInBytes)
{
  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <=",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = setCUDASparseStream();
  CUSPARSE_CHECK(hipsparseXcsrsort_bufferSizeExt(handle, i_m, i_n, i_nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));
}

void Xcsrsort(int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, int *csrColInd, int *P, void *pBuffer)
{
  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = setCUDASparseStream();
  hipsparseMatDescr_t desc;
  hipsparseCreateMatDescr(&desc);
#if TH_INDEX_BASE == 1
  hipsparseSetMatIndexBase(&desc, HIPSPARSE_INDEX_BASE_ONE);
#endif
  CUSPARSE_CHECK(hipsparseXcsrsort(handle, i_m, i_n, i_nnz, desc, csrRowPtr, csrColInd, P, pBuffer));
  // TODO: I think this leaks the matrix descriptor.
}

void Xcoosort_bufferSizeExt(int64_t m, int64_t n, int64_t nnz, const int *cooRows, const int *cooCols, size_t *pBufferSizeInBytes)
{
  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcoosort_bufferSizeExt only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = setCUDASparseStream();
  CUSPARSE_CHECK(hipsparseXcoosort_bufferSizeExt(handle, i_m, i_n, i_nnz, cooRows, cooCols, pBufferSizeInBytes));
}

void XcoosortByRow(int64_t m, int64_t n, int64_t nnz, int *cooRows, int *cooCols, int *P, void *pBuffer)
{
  AT_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "XcoosortByRow only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  auto handle = setCUDASparseStream();
  CUSPARSE_CHECK(hipsparseXcoosortByRow(handle, i_m, i_n, i_nnz, cooRows, cooCols, P, pBuffer));
}


}}}} // namespace at::native::sparse::cuda
