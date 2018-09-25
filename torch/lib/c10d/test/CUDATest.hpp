#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

#include <c10d/CUDAUtils.hpp>

namespace c10d {
namespace test {

void cudaSleep(CUDAStream& stream, uint64_t clocks);

int cudaNumDevices();

} // namespace test
} // namespace c10d
