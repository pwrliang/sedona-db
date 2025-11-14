#pragma once
#include "gpuspatial/utils/cuda_utils.h"
#include "gpuspatial/utils/exception.h"
#include "rmm/cuda_stream_view.hpp"

namespace gpuspatial {
template <typename F, typename... Args>
__global__ void KernelWrapper(F f, Args... args) {
  f(args...);
}

template <typename F, typename... Args>
void LaunchKernel(const rmm::cuda_stream_view& stream, F f, Args&&... args) {
  int grid_size, block_size;

  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                                KernelWrapper<F, Args...>, 0,
                                                reinterpret_cast<int>(MAX_BLOCK_SIZE)));

  KernelWrapper<<<grid_size, block_size, 0, stream>>>(f, std::forward<Args>(args)...);
}

}  // namespace gpuspatial
