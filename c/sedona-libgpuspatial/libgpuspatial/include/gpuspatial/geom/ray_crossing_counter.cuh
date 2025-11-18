
/**********************************************************************
 *
 * GEOS - Geometry Engine Open Source
 * http://geos.osgeo.org
 *
 * Copyright (C) 2006 Refractions Research Inc.
 *
 * This is free software; you can redistribute and/or modify it under
 * the terms of the GNU Lesser General Public Licence as published
 * by the Free Software Foundation.
 * See the COPYING file for more information.
 *
 */
#pragma once
#include "gpuspatial/geom/orientation.cuh"
#include "gpuspatial/geom/point.cuh"
#include "gpuspatial/utils/cuda_utils.h"

namespace gpuspatial {

/** \brief
 * Counts the number of segments crossed by a horizontal ray extending to the
 * right from a given point, in an incremental fashion.
 *
 * This can be used to determine whether a point lies in a polygonal geometry.
 * The class determines the situation where the point lies exactly on a segment.
 * When being used for Point-In-Polygon determination, this case allows
 * short-circuiting the evaluation.
 *
 * This class handles polygonal geometries with any number of shells and holes.
 * The orientation of the shell and hole rings is unimportant.
 * In order to compute a correct location for a given polygonal geometry,
 * it is essential that **all** segments are counted which
 *
 * - touch the ray
 * - lie in in any ring which may contain the point
 *
 * The only exception is when the point-on-segment situation is detected, in
 * which case no further processing is required.
 * The implication of the above rule is that segments which can be a priori
 * determined to *not* touch the ray (i.e. by a test of their bounding box or
 * Y-extent) do not need to be counted. This allows for optimization by indexing.
 *
 * @author Martin Davis
 */

class RayCrossingCounter {
  uint32_t crossing_count_;

  // true if the test point lies on an input segment
  uint32_t point_on_segment_;

 public:
  DEV_HOST_INLINE uint32_t& get_crossing_count() { return crossing_count_; }
  DEV_HOST_INLINE uint32_t& get_point_on_segment() { return point_on_segment_; }

  RayCrossingCounter() = default;

  DEV_HOST RayCrossingCounter(uint32_t crossing_count, uint32_t point_on_segment)
      : crossing_count_(crossing_count), point_on_segment_(point_on_segment) {}

  DEV_HOST_INLINE void Init() {
    crossing_count_ = 0;
    point_on_segment_ = 0;
  }

  /** \brief
   * Counts a segment
   *@param point test point
   * @param p1 an endpoint of the segment
   * @param p2 another endpoint of the segment
   */
  template <typename POINT_T>
  DEV_HOST_INLINE void countSegment(const POINT_T& point, const POINT_T& p1,
                                    const POINT_T& p2) {
    {
      // For each segment, check if it crosses
      // a horizontal ray running from the test point in
      // the positive x direction.

      // check if the segment is strictly to the left of the test point
      if (p1.x() < point.x() && p2.x() < point.x()) {
        return;
      }

      // check if the point is equal to the current ring vertex
      if (point.x() == p2.x() && point.y() == p2.y()) {
        point_on_segment_ = 1;
        return;
      }

      // For horizontal segments, check if the point is on the segment.
      // Otherwise, horizontal segments are not counted.
      if (p1.y() == point.y() && p2.y() == point.y()) {
        double minx = p1.x();
        double maxx = p2.x();

        if (minx > maxx) {
          minx = p2.x();
          maxx = p1.x();
        }

        if (point.x() >= minx && point.x() <= maxx) {
          point_on_segment_ = 1;
        }

        return;
      }

      // Evaluate all non-horizontal segments which cross a horizontal ray
      // to the right of the test pt.
      // To avoid double-counting shared vertices, we use the convention that
      // - an upward edge includes its starting endpoint, and excludes its
      //   final endpoint
      // - a downward edge excludes its starting endpoint, and includes its
      //   final endpoint
      if (((p1.y() > point.y()) && (p2.y() <= point.y())) ||
          ((p2.y() > point.y()) && (p1.y() <= point.y()))) {
        // For an upward edge, orientationIndex will be positive when p1->p2
        // crosses ray. Conversely, downward edges should have negative sign.
        int sign = Orientation<POINT_T>::orientationIndex(p1, p2, point);
        if (sign == 0) {
          point_on_segment_ = 1;
          return;
        }

        if (p2.y() < p1.y()) {
          sign = -sign;
        }

        // The segment crosses the ray if the sign is strictly positive.
        if (sign > 0) {
          crossing_count_++;
        }
      }
    }
  }

  /** \brief
   * Gets the [Location](@ref geom::Location) of the point relative to
   * the ring, polygon or multipolygon from which the processed
   * segments were provided.
   *
   * This method only determines the correct location
   * if **all** relevant segments must have been processed.
   *
   * @return the Location of the point
   */
  DEV_HOST_INLINE PointLocation location() const {
    if (point_on_segment_ == 1) {
      return PointLocation::kBoundary;
    }

    // The point is in the interior of the ring if the number
    // of X-crossings is odd.
    if ((crossing_count_ % 2) == 1) {
      return PointLocation::kInside;
    }

    return PointLocation::kOutside;
  }
};

}  // namespace gpuspatial
