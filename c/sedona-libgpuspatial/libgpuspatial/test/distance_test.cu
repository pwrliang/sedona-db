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

#include "gpuspatial/geom/line_segment.hpp"

#include <gtest/gtest.h>

namespace gpuspatial {
namespace {

template <typename T>
class DistanceTest : public ::testing::Test {};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DistanceTest, ScalarTypes);

// Point-to-Point distance tests.
TYPED_TEST(DistanceTest, PointToPointDistance) {
  using point_t = Point<TypeParam, 2>;
  point_t p1{static_cast<TypeParam>(1), static_cast<TypeParam>(2)};
  point_t p2{static_cast<TypeParam>(4), static_cast<TypeParam>(6)};

  auto d = p1.distance(p2);
  EXPECT_NEAR(d, static_cast<TypeParam>(5), static_cast<TypeParam>(1e-6));
}

// Point-to-LineSegment distance tests.
TYPED_TEST(DistanceTest, PointToSegmentProjectionInside) {
  using point_t = Point<TypeParam, 2>;
  LineSegment<point_t> seg(point_t{static_cast<TypeParam>(0), static_cast<TypeParam>(0)},
                           point_t{static_cast<TypeParam>(4), static_cast<TypeParam>(0)});

  auto d = seg.distance(point_t{static_cast<TypeParam>(2), static_cast<TypeParam>(3)});
  EXPECT_NEAR(d, static_cast<TypeParam>(3), static_cast<TypeParam>(1e-6));
}

TYPED_TEST(DistanceTest, PointToSegmentClampedToStartEndpoint) {
  using point_t = Point<TypeParam, 2>;
  LineSegment<point_t> seg(point_t{static_cast<TypeParam>(0), static_cast<TypeParam>(0)},
                           point_t{static_cast<TypeParam>(4), static_cast<TypeParam>(0)});

  auto d = seg.distance(point_t{static_cast<TypeParam>(-3), static_cast<TypeParam>(4)});
  EXPECT_NEAR(d, static_cast<TypeParam>(5), static_cast<TypeParam>(1e-6));
}

TYPED_TEST(DistanceTest, PointToSegmentClampedToEndEndpoint) {
  using point_t = Point<TypeParam, 2>;
  LineSegment<point_t> seg(point_t{static_cast<TypeParam>(0), static_cast<TypeParam>(0)},
                           point_t{static_cast<TypeParam>(4), static_cast<TypeParam>(0)});

  auto d = seg.distance(point_t{static_cast<TypeParam>(7), static_cast<TypeParam>(4)});
  EXPECT_NEAR(d, static_cast<TypeParam>(5), static_cast<TypeParam>(1e-6));
}

TYPED_TEST(DistanceTest, PointToSegmentZeroForPointOnSegment) {
  using point_t = Point<TypeParam, 2>;
  LineSegment<point_t> seg(point_t{static_cast<TypeParam>(0), static_cast<TypeParam>(0)},
                           point_t{static_cast<TypeParam>(4), static_cast<TypeParam>(0)});

  auto d = seg.distance(point_t{static_cast<TypeParam>(1.5), static_cast<TypeParam>(0)});
  EXPECT_NEAR(d, static_cast<TypeParam>(0), static_cast<TypeParam>(1e-6));
}

TYPED_TEST(DistanceTest, PointToSegmentDegenerateSegmentUsesPointDistance) {
  using point_t = Point<TypeParam, 2>;
  LineSegment<point_t> seg(point_t{static_cast<TypeParam>(1), static_cast<TypeParam>(1)},
                           point_t{static_cast<TypeParam>(1), static_cast<TypeParam>(1)});

  auto d = seg.distance(point_t{static_cast<TypeParam>(4), static_cast<TypeParam>(5)});
  EXPECT_NEAR(d, static_cast<TypeParam>(5), static_cast<TypeParam>(1e-6));
}

// LineSegment-to-LineSegment distance tests.
TYPED_TEST(DistanceTest, SegmentToSegmentIntersectionReturnsZero) {
  using point_t = Point<TypeParam, 2>;
  LineSegment<point_t> ab(point_t{static_cast<TypeParam>(0), static_cast<TypeParam>(0)},
                          point_t{static_cast<TypeParam>(4), static_cast<TypeParam>(4)});
  LineSegment<point_t> cd(point_t{static_cast<TypeParam>(0), static_cast<TypeParam>(4)},
                          point_t{static_cast<TypeParam>(4), static_cast<TypeParam>(0)});

  auto d = ab.distance(cd);
  EXPECT_NEAR(d, static_cast<TypeParam>(0), static_cast<TypeParam>(1e-6));
}

TYPED_TEST(DistanceTest, SegmentToSegmentDisjointParallel) {
  using point_t = Point<TypeParam, 2>;
  LineSegment<point_t> ab(point_t{static_cast<TypeParam>(0), static_cast<TypeParam>(0)},
                          point_t{static_cast<TypeParam>(4), static_cast<TypeParam>(0)});
  LineSegment<point_t> cd(point_t{static_cast<TypeParam>(0), static_cast<TypeParam>(3)},
                          point_t{static_cast<TypeParam>(4), static_cast<TypeParam>(3)});

  auto d = ab.distance(cd);
  EXPECT_NEAR(d, static_cast<TypeParam>(3), static_cast<TypeParam>(1e-6));
}

TYPED_TEST(DistanceTest, SegmentToSegmentDegenerateFirstSegment) {
  using point_t = Point<TypeParam, 2>;
  LineSegment<point_t> ab(point_t{static_cast<TypeParam>(1), static_cast<TypeParam>(2)},
                          point_t{static_cast<TypeParam>(1), static_cast<TypeParam>(2)});
  LineSegment<point_t> cd(point_t{static_cast<TypeParam>(0), static_cast<TypeParam>(0)},
                          point_t{static_cast<TypeParam>(4), static_cast<TypeParam>(0)});

  auto d = ab.distance(cd);
  EXPECT_NEAR(d, static_cast<TypeParam>(2), static_cast<TypeParam>(1e-6));
}

TYPED_TEST(DistanceTest, SegmentToSegmentOverlappingCollinear) {
  using point_t = Point<TypeParam, 2>;
  LineSegment<point_t> ab(point_t{static_cast<TypeParam>(0), static_cast<TypeParam>(0)},
                          point_t{static_cast<TypeParam>(5), static_cast<TypeParam>(0)});
  LineSegment<point_t> cd(point_t{static_cast<TypeParam>(3), static_cast<TypeParam>(0)},
                          point_t{static_cast<TypeParam>(8), static_cast<TypeParam>(0)});

  auto d = ab.distance(cd);
  EXPECT_NEAR(d, static_cast<TypeParam>(0), static_cast<TypeParam>(1e-6));
}

}  // namespace
}  // namespace gpuspatial
