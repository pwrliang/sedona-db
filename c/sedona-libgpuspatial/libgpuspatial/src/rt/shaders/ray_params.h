/*
 * pbrt source code is Copyright(c) 1998-2016
 * Matt Pharr, Greg Humphreys, and Wenzel Jakob.
 *
 * This file is part of pbrt.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "gpuspatial/geom/box.cuh"
#include "gpuspatial/geom/point.cuh"
#include "gpuspatial/utils/cuda_utils.h"

#include <optix.h>
#include <thrust/swap.h>

#include <cfloat>

#define FLT_GAMMA(N) (((N) * FLT_EPSILON) / (1 - (N) * FLT_EPSILON))
#define DBL_GAMMA(N) (((N) * DBL_EPSILON) / (1 - (N) * DBL_EPSILON))
namespace gpuspatial {
namespace detail {

template <int N_DIMS>
struct RayParams {};

template <>
struct RayParams<2> {
  float2 o;  // ray origin
  float2 d;  // ray direction

  DEV_HOST RayParams(const OptixAabb& aabb, bool diagonal) {
    float2 p1{aabb.minX, aabb.minY};
    float2 p2{aabb.maxX, aabb.maxY};

    if (diagonal) {
      p1.x = aabb.maxX;
      p1.y = aabb.minY;
      p2.x = aabb.minX;
      p2.y = aabb.maxY;
    }

    o = p1;
    d = {p2.x - p1.x, p2.y - p1.y};
  }

  DEV_HOST_INLINE void PrintParams(const char* prefix) const {
    printf("%s, o: (%.6f, %.6f), d: (%.6f, %.6f)\n", prefix, o.x, o.y, d.x, d.y);
  }

  DEV_HOST_INLINE bool IsHit(const OptixAabb& aabb) const {
    // FIXME: a little greater than 1.0 as a workaround
    float t0 = 0, t1 = 1.0;  // nextafterf(1.0, FLT_MAX);
    const auto* pMin = reinterpret_cast<const float*>(&aabb.minX);
    const auto* pMax = reinterpret_cast<const float*>(&aabb.maxX);

#pragma unroll
    for (int i = 0; i < 2; ++i) {
      // Update interval for _i_th bounding box slab
      float invRayDir = 1 / reinterpret_cast<const float*>(&d)[i];
      float tNear = (pMin[i] - reinterpret_cast<const float*>(&o)[i]) * invRayDir;
      float tFar = (pMax[i] - reinterpret_cast<const float*>(&o)[i]) * invRayDir;

      // Update parametric interval from slab intersection $t$ values
      if (tNear > tFar) {
        thrust::swap(tNear, tFar);
      }

      // Update _tFar_ to ensure robust ray--bounds intersection
      tFar *= 1 + 2 * FLT_GAMMA(3);
      t0 = tNear > t0 ? tNear : t0;
      t1 = tFar < t1 ? tFar : t1;

      if (t0 > t1) return false;
    }
    return true;
  }
};

}  // namespace detail
}  // namespace gpuspatial
