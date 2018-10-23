#pragma once

#include "THCGeneral.h"

/* Global state of THC. */
struct THCState {
  struct THCRNGState* rngState;
  struct hipDeviceProp_t* deviceProperties;
  /* Set of all allocated resources. blasHandles and sparseHandles do not have
     a default and must be explicitly initialized. We always initialize 1
     blasHandle and 1 sparseHandle but we can use more.
  */
  THCCudaResourcesPerDevice* resourcesPerDevice;
  /* Captured number of devices upon startup; convenience for bounds checking */
  int numDevices;

  /* Allocator using hipHostMalloc. */
  // NB: These allocators (specifically, hipHostAllocator) MUST implement
  // maybeGlobalBoundDeleter, because we have a few use-cases where we need to
  // do raw allocations with them (for Thrust).
  // TODO: Make this statically obvious
  at::Allocator* hipHostAllocator;
  at::Allocator* hipDeviceAllocator;

  /* Table of enabled peer-to-peer access between directed pairs of GPUs.
     If i accessing allocs on j is enabled, p2pAccess[i][j] is 1; 0 otherwise. */
  int** p2pAccessEnabled;
};
