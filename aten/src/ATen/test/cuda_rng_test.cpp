#include "gtest/gtest.h"

#include "ATen/ATen.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime.h"
#include <thread>

void makeRandomNumber() {
  hipSetDevice(std::rand() % 2);
  auto x = at::randn({1000});
}

void testCudaRNGMultithread() {
  auto threads = std::vector<std::thread>();
  for (auto i = 0; i < 1000; i++) {
    threads.emplace_back(makeRandomNumber);
  }
  for (auto& t : threads) {
    t.join();
  }
};

TEST(Cuda_RNGTest, MultithreadRNGTest) {
  testCudaRNGMultithread();
}
