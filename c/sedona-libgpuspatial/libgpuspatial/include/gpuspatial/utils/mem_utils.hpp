#pragma once
#include "gpuspatial/utils/exception.h"

#include "rmm/cuda_stream_view.hpp"

#include <cuda_runtime.h>
namespace gpuspatial {
namespace detail {
template <typename T>
void async_copy_h2d(const rmm::cuda_stream_view& stream, const T* src, T* dst,
                    size_t count) {
  if (count == 0) return;
  // Calculate the total size in bytes from the element count
  size_t size_in_bytes = count * sizeof(T);
  // Issue the asynchronous copy command to the specified stream
  CUDA_CHECK(cudaMemcpyAsync(dst, src, size_in_bytes, cudaMemcpyHostToDevice, stream));
}
template <typename T>
void async_copy_d2h(const rmm::cuda_stream_view& stream, const T* src, T* dst,
                    size_t count) {
  if (count == 0) return;
  // Calculate the total size in bytes from the element count
  size_t size_in_bytes = count * sizeof(T);

  // Issue the asynchronous copy command to the specified stream
  CUDA_CHECK(cudaMemcpyAsync(dst, src, size_in_bytes, cudaMemcpyDeviceToHost, stream));
}
}  // namespace detail
}  // namespace gpuspatial
