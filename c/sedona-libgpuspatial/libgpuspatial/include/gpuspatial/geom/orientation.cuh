/**********************************************************************
 *
 * GEOS - Geometry Engine Open Source
 * http://geos.osgeo.org
 *
 * Copyright (C) 2018 Paul Ramsey <pramsey@cleverlephant.ca>
 *
 * This is free software; you can redistribute and/or modify it under
 * the terms of the GNU Lesser General Public Licence as published
 * by the Free Software Foundation.
 * See the COPYING file for more information.
 *
 **********************************************************************/

#pragma once
#include "gpuspatial/utils/cuda_utils.h"
#include "gpuspatial/utils/dd.h"

namespace gpuspatial {

template <typename POINT_T>
class Orientation {
 public:
  DEV_HOST_INLINE static bool IsCounterClockwise(const ArrayView<POINT_T>& vertices) {
    // # of points without closing endpoint
    int inPts = static_cast<int>(vertices.size()) - 1;
    // sanity check
    if (inPts < 3) return false;

    uint32_t nPts = static_cast<uint32_t>(inPts);
    /**
     * Find first highest point after a lower point, if one exists
     * (e.g. a rising segment)
     * If one does not exist, hiIndex will remain 0
     * and the ring must be flat.
     * Note this relies on the convention that
     * rings have the same start and end point.
     */
    const POINT_T* upHiPt = &vertices[0];
    const POINT_T* upLowPt;

    auto prevY = upHiPt->y();
    uint32_t iUpHi = 0;
    for (uint32_t i = 1; i <= nPts; i++) {
      auto py = vertices[i].y();
      /**
       * If segment is upwards and endpoint is higher, record it
       */
      if (py > prevY && py >= upHiPt->y()) {
        iUpHi = i;
        upHiPt = &vertices[i];
        upLowPt = &vertices[i - 1];
      }
      prevY = py;
    }
    /**
     * Check if ring is flat and return default value if so
     */
    if (iUpHi == 0) return false;

    /**
     * Find the next lower point after the high point
     * (e.g. a falling segment).
     * This must exist since ring is not flat.
     */
    uint32_t iDownLow = iUpHi;
    do {
      iDownLow = (iDownLow + 1) % nPts;
    } while (iDownLow != iUpHi && vertices[iDownLow].y() == upHiPt->y());

    const auto& downLowPt = vertices[iDownLow];
    uint32_t iDownHi = iDownLow > 0 ? iDownLow - 1 : nPts - 1;
    const auto& downHiPt = vertices[iDownHi];

    /**
     * Two cases can occur:
     * 1) the hiPt and the downPrevPt are the same.
     *    This is the general position case of a "pointed cap".
     *    The ring orientation is determined by the orientation of the cap
     * 2) The hiPt and the downPrevPt are different.
     *    In this case the top of the cap is flat.
     *    The ring orientation is given by the direction of the flat segment
     */
    if (*upHiPt == downHiPt) {
      /**
       * Check for the case where the cap has configuration A-B-A.
       * This can happen if the ring does not contain 3 distinct points
       * (including the case where the input array has fewer than 4 elements), or
       * it contains coincident line segments.
       */
      if (*upLowPt == *upHiPt || downLowPt == *upHiPt || *upLowPt == downLowPt)
        return false;

      /**
       * It can happen that the top segments are coincident.
       * This is an invalid ring, which cannot be computed correctly.
       * In this case the orientation is 0, and the result is false.
       */
      return orientationIndex(*upLowPt, *upHiPt, downLowPt) == COUNTERCLOCKWISE;
    } else {
      /**
       * Flat cap - direction of flat top determines orientation
       */
      double delX = downHiPt.x() - upHiPt->x();
      return delX < 0;
    }
  }
  DEV_HOST_INLINE static int orientationIndex(const POINT_T& p1, const POINT_T& p2,
                                              const POINT_T& q) {
    return orientationIndex(p1.x(), p1.y(), p2.x(), p2.y(), q.x(), q.y());
  }

 private:
  enum { CLOCKWISE = -1, COLLINEAR = 0, COUNTERCLOCKWISE = 1 };

  enum { RIGHT = -1, LEFT = 1, STRAIGHT = 0, FAILURE = 2 };

  DEV_HOST_INLINE static int orientation(double x) {
    if (x < 0) {
      return RIGHT;
    }
    if (x > 0) {
      return LEFT;
    }
    return STRAIGHT;
  };

  DEV_HOST_INLINE static int orientationIndexFilter(double pax, double pay, double pbx,
                                                    double pby, double pcx, double pcy) {
    /**
     * A value which is safely greater than the relative round-off
     * error in double-precision numbers
     */
    double constexpr DP_SAFE_EPSILON = 1e-15;

    double detsum;
    double const detleft = (pax - pcx) * (pby - pcy);
    double const detright = (pay - pcy) * (pbx - pcx);
    double const det = detleft - detright;

    if (detleft > 0.0) {
      if (detright <= 0.0) {
        return orientation(det);
      } else {
        detsum = detleft + detright;
      }
    } else if (detleft < 0.0) {
      if (detright >= 0.0) {
        return orientation(det);
      } else {
        detsum = -detleft - detright;
      }
    } else {
      return orientation(det);
    }

    double const errbound = DP_SAFE_EPSILON * detsum;
    if ((det >= errbound) || (-det >= errbound)) {
      return orientation(det);
    }
    return FAILURE;
  };

  DEV_HOST_INLINE static int OrientationDD(const DD& dd) {
    DD const zero(0.0);
    if (dd < zero) {
      return RIGHT;
    }

    if (dd > zero) {
      return LEFT;
    }

    return STRAIGHT;
  }

  DEV_HOST_INLINE static int orientationIndex(double p1x, double p1y, double p2x,
                                              double p2y, double qx, double qy) {
    if (!std::isfinite(qx) || !std::isfinite(qy)) {
      assert(false);
    }

    // fast filter for orientation index
    // avoids use of slow extended-precision arithmetic in many cases
    int index = orientationIndexFilter(p1x, p1y, p2x, p2y, qx, qy);
    if (index <= 1) {
      return index;
    }

    // normalize coordinates
    DD dx1 = DD(p2x) + DD(-p1x);
    DD dy1 = DD(p2y) + DD(-p1y);
    DD dx2 = DD(qx) + DD(-p2x);
    DD dy2 = DD(qy) + DD(-p2y);

    // sign of determinant - inlined for performance
    DD mx1y2(dx1 * dy2);
    DD my1x2(dy1 * dx2);
    DD d = mx1y2 - my1x2;
    return OrientationDD(d);
  }
};
}  // namespace gpuspatial

