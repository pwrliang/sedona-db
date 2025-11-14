#pragma once

#include <string>
#include <type_traits>

namespace gpuspatial {
namespace detail {

template <typename POINT_T>
std::string GetShaderPointTypeId() {
  using scalar_t = typename POINT_T::scalar_t;
  constexpr int n_dim = POINT_T::n_dim;

  // Use `if constexpr` for compile-time branching instead of runtime `typeid`.
  const char* type;
  if constexpr (std::is_same_v<scalar_t, float>) {
    type = "FLOAT";
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    type = "DOUBLE";
  } else {
    // Fail at compile time for unsupported types with a clear error message.
    static_assert(std::is_same_v<scalar_t, void>,
                  "Unsupported point scalar type. Only float or double are allowed.");
  }

  const char* nd;
  if constexpr (n_dim == 2) {
    nd = "2D";
  } else if constexpr (n_dim == 3) {
    nd = "3D";
  } else {
    // Fail at compile time for unsupported dimensions.
    static_assert(n_dim == 0, "Unsupported point dimension. Only 2 or 3 are allowed.");
  }

  // Use safe C++ string concatenation, avoiding unsafe C-style `sprintf`.
  return std::string("SHADER_POINT_") + type + "_" + nd;
}

template <typename INDEX_T>
std::string GetShaderIndexTypeId() {
  if constexpr (std::is_same_v<INDEX_T, uint32_t>) {
    return "SHADER_INDEX_UINT32";
  } else if constexpr (std::is_same_v<INDEX_T, uint64_t>) {
    return "SHADER_INDEX_UINT64";
  } else {
    // Fail at compile time for unsupported index types.
    static_assert(std::is_same_v<INDEX_T, void>,
                  "Unsupported index type. Only uint32_t or uint64_t are allowed.");
  }
}
}  // namespace detail

template <typename POINT_T>
inline std::string GetPointQueryShaderId() {
  return detail::GetShaderPointTypeId<POINT_T>() + "_point_query.ptx";
}

template <typename POINT_T>
inline std::string GetBoxQueryForwardShaderId() {
  return detail::GetShaderPointTypeId<POINT_T>() + "_box_query_forward.ptx";
}

template <typename POINT_T>
inline std::string GetBoxQueryBackwardShaderId() {
  return detail::GetShaderPointTypeId<POINT_T>() + "_box_query_backward.ptx";
}

template <typename POINT_T>
inline std::string GetPolygonPointQueryShaderId() {
  return detail::GetShaderPointTypeId<POINT_T>() + "_polygon_point_query.ptx";
}

template <typename POINT_T>
inline std::string GetMultiPolygonPointQueryShaderId() {
  return detail::GetShaderPointTypeId<POINT_T>() + "_multipolygon_point_query.ptx";
}
}  // namespace gpuspatial
