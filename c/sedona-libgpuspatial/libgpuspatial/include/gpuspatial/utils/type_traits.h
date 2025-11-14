#pragma once
#include <vector_types.h>

namespace gpuspatial {
template <typename SCALA_T, int N_DIM>
struct cuda_vec {};

template <>
struct cuda_vec<float, 2> {
  using type = float2;
};

template <>
struct cuda_vec<float, 3> {
  using type = float3;
};

template <>
struct cuda_vec<double, 2> {
  using type = double2;
};

template <>
struct cuda_vec<double, 3> {
  using type = double3;
};

template <typename CUDA_VEC_T>
struct cuda_vec_info {};

template <>
struct cuda_vec_info<float2> {
  using scalar_type = float;
  static constexpr int n_dim = 2;
};

template <>
struct cuda_vec_info<float3> {
  using scalar_type = float;
  static constexpr int n_dim = 3;
};

template <>
struct cuda_vec_info<double2> {
  using scalar_type = double;
  static constexpr int n_dim = 2;
};

template <>
struct cuda_vec_info<double3> {
  using scalar_type = double;
  static constexpr int n_dim = 3;
};
}  // namespace gpuspatial
