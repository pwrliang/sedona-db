// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
#pragma once
#include "gpuspatial/geom/box.hpp"
#include "gpuspatial/geom/point.hpp"
#include "gpuspatial/utils/cuda_utils.hpp"
#include "gpuspatial/utils/floating_point.hpp"

#include <cmath>

namespace gpuspatial {
template <typename POINT_T>
class LineSegment {
  using point_t = POINT_T;
  using scalar_t = typename point_t::scalar_t;
  static constexpr int n_dim = point_t::n_dim;
  using box_t = Box<point_t>;

 public:
  LineSegment() = default;
  DEV_HOST LineSegment(const point_t& p1, const point_t& p2) : p1_(p1), p2_(p2) {}

  DEV_HOST_INLINE const point_t& get_p1() const { return p1_; }

  DEV_HOST_INLINE const point_t& get_p2() const { return p2_; }

  DEV_HOST_INLINE point_t centroid() const {
    point_t c;
    for (int i = 0; i < n_dim; i++) {
      c.set_coordinate(i, (p1_.get_coordinate(i) + p2_.get_coordinate(i)) / 2.0);
    }
    return c;
  }

  DEV_HOST_INLINE int orientation(const point_t& q) const {
    auto d_x = (q.x() - p1_.x());
    auto d_y = (q.y() - p1_.y());
    typename point_t::scalar_t constexpr zero = 0.0;

    if (float_equal(d_x, zero) && float_equal(d_y, zero)) {
      return 0;
    }
    auto v1 = d_x * (p2_.y() - p1_.y());
    auto v2 = (p2_.x() - p1_.x()) * d_y;

    if (float_equal(v1, v2)) {
      return 0;
    }
    auto side = v1 - v2;
    return side < 0 ? -1 : 1;
  }

  DEV_HOST_INLINE box_t get_mbr() const {
    point_t min_p, max_p;
    for (int dim = 0; dim < n_dim; dim++) {
      min_p.set_coordinate(dim, std::numeric_limits<scalar_t>::max());
      max_p.set_coordinate(dim, std::numeric_limits<scalar_t>::lowest());
    }

    for (int dim = 0; dim < n_dim; dim++) {
      auto v1 = p1_.get_coordinate(dim);
      auto v2 = p2_.get_coordinate(dim);
      auto min_v = std::min(v1, v2);
      auto max_v = std::max(v1, v2);
      min_p.set_coordinate(dim, std::min(min_p.get_coordinate(dim), min_v));
      max_p.set_coordinate(dim, std::max(max_p.get_coordinate(dim), max_v));
    }
    return box_t(min_p, max_p);
  }

  template <typename point_type = POINT_T,
            typename std::enable_if<point_type::n_dim == 2, bool>::type = true>
  DEV_HOST_INLINE bool covers(const point_type& q) const {
    auto side = ((q.x() - p1_.x()) * (p2_.y() - p1_.y()) -
                 (p2_.x() - p1_.x()) * (q.y() - p1_.y()));

    if (side == 0) {
      return (p1_.x() <= q.x() && q.x() <= p2_.x()) ||
             (p1_.x() >= q.x() && q.x() >= p2_.x()) ||
             (p1_.y() <= q.y() && q.y() <= p2_.y()) ||
             (p1_.y() >= q.y() && q.y() >= p2_.y());
    }
    return false;
  }

  template <typename point_type = POINT_T,
            typename std::enable_if<point_type::n_dim == 2, bool>::type = true>
  DEV_HOST_INLINE PointLocation locate_point(const point_t& q) const {
    if (orientation(q) == 0) {
      if (((p1_.x() <= q.x() && q.x() <= p2_.x()) ||
           (p2_.x() <= q.x() && q.x() <= p1_.x())) &&
          ((p1_.y() <= q.y() && q.y() <= p2_.y()) ||
           (p2_.y() <= q.y() && q.y() <= p1_.y()))) {
        if ((p1_.x() == q.x() && p1_.y() == q.y()) ||
            (p2_.x() == q.x() && p2_.y() == q.y()))
          return PointLocation::kBoundary;
        return PointLocation::kInside;
      }
    }

    return PointLocation::kOutside;
  }

  template <typename point_type = POINT_T,
            typename std::enable_if<point_type::n_dim == 2, bool>::type = true>
  DEV_HOST_INLINE scalar_t distance(const point_type& q) const {
    point_t const q_pt{q.x(), q.y()};
    auto const ab_x = p2_.x() - p1_.x();
    auto const ab_y = p2_.y() - p1_.y();
    auto const aq_x = q_pt.x() - p1_.x();
    auto const aq_y = q_pt.y() - p1_.y();
    auto const ab_len_sq = ab_x * ab_x + ab_y * ab_y;
    scalar_t constexpr zero = 0.0;

    // Degenerate segment: fall back to point-to-point distance.
    if (float_equal(ab_len_sq, zero)) {
      return p1_.distance(q_pt);
    }

    // Project q onto line AB using normalized dot product.
    auto const r = (aq_x * ab_x + aq_y * ab_y) / ab_len_sq;

    // Projection before A: closest point is A.
    if (r <= zero) {
      return p1_.distance(q_pt);
    }

    // Projection after B: closest point is B.
    if (r >= scalar_t{1}) {
      return p2_.distance(q_pt);
    }

    // Projection lies on segment interior.
    point_t const proj{p1_.x() + r * ab_x, p1_.y() + r * ab_y};
    return q_pt.distance(proj);
  }

  template <typename point_type = POINT_T,
            typename std::enable_if<point_type::n_dim == 2, bool>::type = true>
  DEV_HOST_INLINE scalar_t distance(const LineSegment<point_type>& other) const {
    scalar_t constexpr zero = 0.0;
    scalar_t constexpr one = 1.0;

    // Degenerate segment: reduce to point-to-segment distance.
    if (p1_ == p2_) {
      return other.distance(p1_);
    }

    if (other.get_p1() == other.get_p2()) {
      return distance(other.get_p1());
    }

    bool no_intersection = false;

    // Fast reject: disjoint bounding boxes cannot intersect.
    if (!get_mbr().intersects(other.get_mbr())) {
      no_intersection = true;
    } else {
      auto const& a = p1_;
      auto const& b = p2_;
      auto const& c = other.get_p1();
      auto const& d = other.get_p2();

      auto const denom =
          (b.x() - a.x()) * (d.y() - c.y()) - (b.y() - a.y()) * (d.x() - c.x());

      // Parallel (including collinear) lines are handled by endpoint fallback.
      if (float_equal(denom, zero)) {
        no_intersection = true;
      } else {
        auto const r_num =
            (a.y() - c.y()) * (d.x() - c.x()) - (a.x() - c.x()) * (d.y() - c.y());
        auto const s_num =
            (a.y() - c.y()) * (b.x() - a.x()) - (a.x() - c.x()) * (b.y() - a.y());

        auto const r = r_num / denom;
        auto const s = s_num / denom;

        no_intersection = (r < zero) || (r > one) || (s < zero) || (s > one);
      }
    }

    if (no_intersection) {
      // Minimum of endpoint-to-opposite-segment distances.
      return std::min(distance(other.get_p1()),
                      std::min(distance(other.get_p2()),
                               std::min(other.distance(p1_), other.distance(p2_))));
    }

    return zero;
  }

 private:
  point_t p1_, p2_;
};

}  // namespace gpuspatial
