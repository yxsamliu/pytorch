#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include <atomic>
#include <mutex>

typedef struct THCGeneratorState {
  hiprandStateMtgp32_t* gen_states;
  mtgp32_kernel_params_t *kernel_params;
  int initf;
  uint64_t initial_seed;
  std::atomic<int64_t> philox_seed_offset;
} THCGeneratorState;

struct THCGenerator {
  std::mutex mutex; /* mutex for using this generator */
  THCGeneratorState state;
};
