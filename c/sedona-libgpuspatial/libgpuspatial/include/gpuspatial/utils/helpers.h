#ifndef GPUSPATIAL_UTILS_HELPERS_H
#define GPUSPATIAL_UTILS_HELPERS_H
#include <cassert>
#include <climits>
#include <cmath>
#include <limits>
#include <type_traits>

#include "gpuspatial/utils/cuda_utils.h"

namespace gpuspatial {
// Adapted from
// http://stackoverflow.com/questions/466204/rounding-up-to-nearest-power-of-2
template <typename UnsignedType>
DEV_HOST_INLINE UnsignedType next_power_2(UnsignedType v) {
  static_assert(std::is_unsigned<UnsignedType>::value, "Only works for unsigned types");
  --v;
  for (int i = 1; i < sizeof(v) * CHAR_BIT; i *= 2) {
    v |= v >> i;
  }
  return ++v;
}

/**
 * Convert double to float with rounding
 *
 * @v double
 * @dir dir = -1, round down; dir = 1 round up
 * @iter number of calling nextafter
 */
DEV_HOST_INLINE float next_float_from_double(double v, int dir, int iter = 1) {
  assert(dir == 1 || dir == -1);
  auto fv = static_cast<float>(v);  // pos number
  if (fv == 0) {
    return 0.0f;
  }
  float to = v * dir < 0 ? 0 : dir * std::numeric_limits<float>::infinity();

  for (int i = 0; i < iter; i++) {
    fv = std::nextafter(fv, to);
  }

  return fv;
}

template <typename T>
DEV_HOST_INLINE void unpack64(unsigned int i0, unsigned int i1, T* t) {
  static_assert(sizeof(T) == 8, "Unpacking an invalid Type");
  *reinterpret_cast<unsigned long long*>(t) =
      static_cast<unsigned long long>(i0) << 32 | i1;
}

template <typename T>
DEV_HOST_INLINE void pack64(T* t, unsigned int& i0, unsigned int& i1) {
  static_assert(sizeof(T) == 8, "Packing an invalid Type");
  const unsigned long long ud = *reinterpret_cast<unsigned long long*>(t);
  i0 = ud >> 32;
  i1 = ud & 0x00000000ffffffff;
}

}  // namespace gpuspatial

#endif  // GPUSPATIAL_UTILS_HELPERS_H
