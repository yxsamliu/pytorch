#include "THCGeneral.h"
#include "TH.h"
#include "THCAllocator.h"
#include "THCCachingHostAllocator.h"
#include "THCTensorRandom.h"
#include "THCGeneral.hpp"

#include "ATen/cuda/CUDAStream.h"

#include "THCCachingAllocator.h"
#include <stdlib.h>
#include <stdint.h>

/* Size of scratch space available in global memory per each SM + stream */
#define MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM 4 * sizeof(float)

/* Minimum amount of scratch space per device. Total scratch memory per
 * device is either this amount, or the # of SMs * the space per SM defined
 * above, whichever is greater.*/
#define MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE 32768 * sizeof(float)

/* Maximum number of P2P connections (if there are more than 9 then P2P is
 * enabled in groups of 8). */
#define THC_CUDA_MAX_PEER_SIZE 8

void THCState_free(THCState* state)
{
  free(state);
}

THCCudaResourcesPerDevice* THCState_getDeviceResourcePtr(
  THCState *state, int device);

THCState* THCState_alloc(void)
{
  THCState* state = (THCState*) malloc(sizeof(THCState));
  memset(state, 0, sizeof(THCState));
  return state;
}

void THCudaInit(THCState* state)
{
  if (!state->hipDeviceAllocator) {
    state->hipDeviceAllocator = THCCachingAllocator_get();
  }
  if (!state->hipHostAllocator) {
    state->hipHostAllocator = getTHCCachingHostAllocator();
  }

  int numDevices = 0;
  THCudaCheck(hipGetDeviceCount(&numDevices));
  state->numDevices = numDevices;

  int device = 0;
  THCudaCheck(hipGetDevice(&device));

  state->resourcesPerDevice = (THCCudaResourcesPerDevice*)
    malloc(numDevices * sizeof(THCCudaResourcesPerDevice));
  memset(state->resourcesPerDevice, 0, numDevices * sizeof(THCCudaResourcesPerDevice));

  state->deviceProperties =
    (struct hipDeviceProp_t*)malloc(numDevices * sizeof(struct hipDeviceProp_t));

  state->rngState = (THCRNGState*)malloc(sizeof(THCRNGState));
  THCRandom_init(state, numDevices, device);

  // p2pAccessEnabled records if p2p copies are allowed between pairs of
  // devices. Values include "1" (copy allowed), "0" (copy not allowed), and
  // "-1" (unknown).
  // Currently the max number of gpus in P2P group is 8, so if there are more
  // we enable P2P in groups of 8
  state->p2pAccessEnabled = (int**) malloc(sizeof(int*) * numDevices);
  for (int i = 0; i < numDevices; ++i) {
    state->p2pAccessEnabled[i] = (int*) malloc(sizeof(int) * numDevices);
    for (int j = 0; j < numDevices; ++j)
      if (i == j)
        state->p2pAccessEnabled[i][j] = 1;
      else if (j / THC_CUDA_MAX_PEER_SIZE != i / THC_CUDA_MAX_PEER_SIZE)
        state->p2pAccessEnabled[i][j] = 0;
      else
        state->p2pAccessEnabled[i][j] = -1;
  }

  for (int i = 0; i < numDevices; ++i) {
    THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, i);
    THCudaCheck(hipSetDevice(i));
    THCudaCheck(hipGetDeviceProperties(&state->deviceProperties[i], i));

    /* The scratch space that we want to have available per each device is
       based on the number of SMs available per device. We guarantee a
       minimum of 128kb of space per device, but to future-proof against
       future architectures that may have huge #s of SMs, we guarantee that
       we have at least 16 bytes for each SM. */
    int numSM = state->deviceProperties[i].multiProcessorCount;
    size_t sizePerStream =
      MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE >= numSM * MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM ?
      MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE :
      numSM * MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM;
    res->scratchSpacePerStream = sizePerStream;
  }

  /* Restore to previous device */
  THCudaCheck(hipSetDevice(device));
}

void THCudaShutdown(THCState* state)
{
  THCRandom_shutdown(state);

  free(state->rngState);
  free(state->deviceProperties);

  int deviceCount = 0;
  int prevDev = -1;
  THCudaCheck(hipGetDevice(&prevDev));
  THCudaCheck(hipGetDeviceCount(&deviceCount));

  /* cleanup p2p access state */
  for (int dev = 0; dev < deviceCount; ++dev) {
    free(state->p2pAccessEnabled[dev]);
  }
  free(state->p2pAccessEnabled);

  /* cleanup per-device state */
  for (int dev = 0; dev < deviceCount; ++dev) {
    THCudaCheck(hipSetDevice(dev));
    THCCudaResourcesPerDevice* res = &(state->resourcesPerDevice[dev]);

    // Frees BLAS handle
    if (res->blasHandle) {
      THCublasCheck(rocblas_destroy_handle(res->blasHandle));
    }

    // Frees sparse handle
    if (res->sparseHandle) {
      THCusparseCheck(hipsparseDestroy(res->sparseHandle));
    }
  }

  free(state->resourcesPerDevice);
  if (state->hipDeviceAllocator == THCCachingAllocator_get()) {
    THCCachingAllocator_emptyCache();
  }
  if (state->hipHostAllocator == getTHCCachingHostAllocator()) {
    THCCachingHostAllocator_emptyCache();
  }

  THCudaCheck(hipSetDevice(prevDev));
}

int THCState_getPeerToPeerAccess(THCState* state, int dev, int devToAccess)
{
  if (dev < 0 || dev >= state->numDevices) {
    THError("%d is not a device", dev);
  }
  if (devToAccess < 0 || devToAccess >= state->numDevices) {
    THError("%d is not a device", devToAccess);
  }
  if (state->p2pAccessEnabled[dev][devToAccess] == -1) {
    int prevDev = 0;
    THCudaCheck(hipGetDevice(&prevDev));
    THCudaCheck(hipSetDevice(dev));

    int access = 0;
    THCudaCheck(hipDeviceCanAccessPeer(&access, dev, devToAccess));
    if (access) {
      hipError_t err = hipDeviceEnablePeerAccess(devToAccess, 0);
      if (err == hipErrorPeerAccessAlreadyEnabled) {
        // ignore and clear the error if access was already enabled
        hipGetLastError();
      } else {
        THCudaCheck(err);
      }
      state->p2pAccessEnabled[dev][devToAccess] = 1;
    } else {
      state->p2pAccessEnabled[dev][devToAccess] = 0;
    }

    THCudaCheck(hipSetDevice(prevDev));
  }
  return state->p2pAccessEnabled[dev][devToAccess];
}

struct hipDeviceProp_t* THCState_getCurrentDeviceProperties(THCState* state)
{
  int curDev = -1;
  THCudaCheck(hipGetDevice(&curDev));

  return &(state->deviceProperties[curDev]);
}

struct hipDeviceProp_t* THCState_getDeviceProperties(THCState* state, int device)
{
  THAssert(device >= 0 && device < state->numDevices);
  return &(state->deviceProperties[device]);
}

struct THCRNGState* THCState_getRngState(THCState *state)
{
  return state->rngState;
}

THAllocator* THCState_getCudaHostAllocator(THCState* state)
{
  return state->hipHostAllocator;
}

int THCState_getNumDevices(THCState *state)
{
  return state->numDevices;
}

THCCudaResourcesPerDevice* THCState_getDeviceResourcePtr(
  THCState *state, int device)
{
  /* `device` is a CUDA index */
  if (device >= state->numDevices || device < 0)
  {
    THError("%d is not a device", device + 1 /* back to Torch index */);
  }

  return &(state->resourcesPerDevice[device]);
}

THCStream* THCState_getStreamOnDevice(THCState* state, int device) {
  return at::cuda::detail::CUDAStream_getCurrentStream(device);
}

void THCState_setStreamOnDevice(THCState *state, int device, THCStream *stream) {
  at::cuda::detail::CUDAStream_setStream(stream);
}

THC_API void THCState_setStream(THCState *state, THCStream* stream) {
  at::cuda::detail::CUDAStream_setStream(stream);
}

hipStream_t THCState_getCurrentStreamOnDevice(THCState *state, int device) {
  return at::cuda::detail::CUDAStream_stream(
    at::cuda::detail::CUDAStream_getCurrentStream(device));
}

hipStream_t THCState_getCurrentStream(THCState *state) {
  return at::cuda::detail::CUDAStream_stream(
    at::cuda::detail::CUDAStream_getCurrentStream());
}

THCStream* THCState_getStream(THCState *state) {
  return at::cuda::detail::CUDAStream_getCurrentStream();
}

rocblas_handle THCState_getCurrentBlasHandle(THCState *state)
{
  // Short-circuits if state is NULL
  // Note: possible in debugging code or improperly instrumented kernels
  if (!state) {
    THError("THCState and sparseHandles must be set as there is no default sparseHandle");
    return NULL;
  }

  int device;
  THCudaCheck(hipGetDevice(&device));

  // Creates the BLAS handle if not created yet
  THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  if (!res->blasHandle) {
    THCublasCheck(rocblas_create_handle(&res->blasHandle));
  }

  return res->blasHandle;
}

hipsparseHandle_t THCState_getCurrentSparseHandle(THCState *state)
{
  // Short-circuits if state is NULL
  // Note: possible in debugging code or improperly instrumented kernels
  if (!state) {
    THError("THCState and sparseHandles must be set as there is no default sparseHandle");
    return NULL;
  }

  int device;
  THCudaCheck(hipGetDevice(&device));

  // Creates the sparse handle if not created yet
  THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  if (!res->sparseHandle) {
    THCusparseCheck(hipsparseCreate(&res->sparseHandle));
  }

  return res->sparseHandle;
}

size_t THCState_getCurrentDeviceScratchSpaceSize(THCState* state)
{
  int device = -1;
  THCudaCheck(hipGetDevice(&device));
  THCCudaResourcesPerDevice* res = THCState_getDeviceResourcePtr(state, device);
  return res->scratchSpacePerStream;
}

void __THCudaCheck(hipError_t err, const char *file, const int line)
{
  if(err != hipSuccess)
  {
    static int alreadyFailed = 0;
    if(!alreadyFailed) {
      fprintf(stderr, "THCudaCheck FAIL file=%s line=%i error=%i : %s\n", file, line, err, hipGetErrorString(err));
      alreadyFailed = 1;
    }
    _THError(file, line, "cuda runtime error (%d) : %s", err,
             hipGetErrorString(err));
  }
}

void __THCudaCheckWarn(hipError_t err, const char *file, const int line)
{
  if(err != hipSuccess)
  {
    fprintf(stderr, "THCudaCheckWarn FAIL file=%s line=%i error=%i : %s\n", file, line, err, hipGetErrorString(err));
  }
}

void __THCublasCheck(rocblas_status status, const char *file, const int line)
{
  if(status != rocblas_status_success)
  {
    const char* errmsg = NULL;

    switch(status)
    {
      case rocblas_status_invalid_handle:
        errmsg = "library not initialized";
        break;

      case rocblas_status_memory_error:
        errmsg = "resource allocation failed";
        break;

      case rocblas_status_invalid_pointer:
        errmsg = "an invalid numeric value was used as an argument";
        break;

      case rocblas_status_not_implemented:
        errmsg = "an absent device architectural feature is required";
        break;

#ifndef __HIP_PLATFORM_HCC__
      case rocblas_status_internal_error:
        errmsg = "an access to GPU memory space failed";
        break;

      case rocblas_status_internal_error:
        errmsg = "the GPU program failed to execute";
        break;
#endif

      case rocblas_status_internal_error:
        errmsg = "an internal operation failed";
        break;

      default:
        errmsg = "unknown error";
        break;
    }

    _THError(file, line, "cublas runtime error : %s", errmsg);
  }
}

void __THCusparseCheck(hipsparseStatus_t status, const char *file, const int line)
{
  if(status != HIPSPARSE_STATUS_SUCCESS)
  {
    const char* errmsg = NULL;

    switch(status)
    {
      case HIPSPARSE_STATUS_NOT_INITIALIZED:
        errmsg = "library not initialized";
        break;

      case HIPSPARSE_STATUS_ALLOC_FAILED:
        errmsg = "resource allocation failed";
        break;

      case HIPSPARSE_STATUS_INVALID_VALUE:
        errmsg = "an invalid numeric value was used as an argument";
        break;

      case HIPSPARSE_STATUS_ARCH_MISMATCH:
        errmsg = "an absent device architectural feature is required";
        break;

      case HIPSPARSE_STATUS_MAPPING_ERROR:
        errmsg = "an access to GPU memory space failed";
        break;

      case HIPSPARSE_STATUS_EXECUTION_FAILED:
        errmsg = "the GPU program failed to execute";
        break;

      case HIPSPARSE_STATUS_INTERNAL_ERROR:
        errmsg = "an internal operation failed";
        break;

      case HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        errmsg = "the matrix type is not supported by this function";
        break;

      default:
        errmsg = "unknown error";
        break;
    }

    _THError(file, line, "cusparse runtime error : %s", errmsg);
  }
}

void* THCudaMalloc(THCState *state, size_t size)
{
  THCudaCheck(hipGetLastError());
  THCDeviceAllocator* allocator = state->hipDeviceAllocator;
  return allocator->raw_allocate(size);
}

void THCudaFree(THCState *state, void* ptr) {
  state->hipDeviceAllocator->raw_deallocate(ptr);
}

at::DataPtr THCudaHostAlloc(THCState *state, size_t size)
{
  THCudaCheck(hipGetLastError());
  THAllocator* allocator = state->hipHostAllocator;
  return allocator->allocate(size);
}

void THCudaHostRecord(THCState *state, void *ptr) {
  if (state->hipHostAllocator == getTHCCachingHostAllocator()) {
    THCStream* stream = THCState_getStream(state);
    THCCachingHostAllocator_recordEvent(ptr, stream);
  }
}

hipError_t THCudaMemGetInfo(THCState *state,  size_t* freeBytes, size_t* totalBytes, size_t* largestBlock)
{
  size_t cachedBytes = 0;
  THCDeviceAllocator* allocator = state->hipDeviceAllocator;

  *largestBlock = 0;
  /* get info from CUDA first */
  hipError_t ret = hipMemGetInfo(freeBytes, totalBytes);
  if (ret!= hipSuccess)
    return ret;

  int device;
  ret = hipGetDevice(&device);
  if (ret!= hipSuccess)
    return ret;

  /* not always true - our optimistic guess here */
  *largestBlock = *freeBytes;

  if (allocator == THCCachingAllocator_get()) {
    THCCachingAllocator_cacheInfo(device, &cachedBytes, largestBlock);
  }

  /* Adjust resulting free bytes number. largesBlock unused for now */
  *freeBytes += cachedBytes;
  return hipSuccess;
}

#undef MIN_GLOBAL_SCRATCH_SPACE_PER_SM_STREAM
#undef MIN_GLOBAL_SCRATCH_SPACE_PER_DEVICE

#include "THCStorage.cpp"
#include "THCAllocator.cpp"
