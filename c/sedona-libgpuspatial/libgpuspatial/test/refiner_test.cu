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
#include "array_stream.hpp"
#include "test_common.hpp"

#include "gpuspatial/index/rt_spatial_index.hpp"
#include "gpuspatial/loader/device_geometries.hpp"
#include "gpuspatial/refine/rt_spatial_refiner.hpp"

#include "nanoarrow/nanoarrow.hpp"

#include <geos/geom/prep/PreparedGeometry.h>
#include <geos/geom/prep/PreparedGeometryFactory.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <chrono>
#include <iomanip>
#include <thread>

#include "gpuspatial/index/rt_spatial_index.cuh"
#include "gpuspatial/refine/rt_spatial_refiner.cuh"

namespace gpuspatial {

void TestJoiner(ArrowSchema* build_schema, std::vector<ArrowArray*>& build_arrays,
                ArrowSchema* probe_schema, std::vector<ArrowArray*>& probe_arrays,
                Predicate predicate, bool pipelined = false) {
  using namespace TestUtils;
  using coord_t = double;
  using fpoint_t = Point<coord_t, 2>;
  using box_t = Box<fpoint_t>;

  auto rt_engine = std::make_shared<RTEngine>();
  {
    std::string ptx_root = TestUtils::GetTestShaderPath();
    auto config = get_default_rt_config(ptx_root);
    rt_engine->Init(config);
  }

  RTSpatialIndexConfig idx_config;
  idx_config.rt_engine = rt_engine;
  auto rt_index = CreateRTSpatialIndex<coord_t, 2>(idx_config);

  RTSpatialRefinerConfig refiner_config;
  refiner_config.rt_engine = rt_engine;
  if (pipelined) {
    refiner_config.pipeline_batches = 10;
  }
  auto rt_refiner = CreateRTSpatialRefiner(refiner_config);

  // Initialize GEOS C++ components
  auto geos_factory = geos::geom::GeometryFactory::create();
  geos::io::WKBReader wkb_reader(*geos_factory);
  geos::index::strtree::STRtree tree(10);

  // Storage for GEOS geometries to ensure they outlive the tree
  // The STRtree stores raw pointers, so we must own the objects
  std::vector<std::unique_ptr<geos::geom::Geometry>> build_geoms_storage;
  size_t total_build_length = 0;
  for (auto& array : build_arrays) {
    total_build_length += array->length;
  }
  build_geoms_storage.reserve(total_build_length);

  size_t tail_build = 0;
  ArrowError error;

  // --- Build Phase ---
  for (auto& array : build_arrays) {
    nanoarrow::UniqueArrayView array_view;
    ASSERT_EQ(ArrowArrayViewInitFromSchema(array_view.get(), build_schema, &error),
              NANOARROW_OK)
        << error.message;
    ASSERT_EQ(ArrowArrayViewSetArray(array_view.get(), array, &error), NANOARROW_OK)
        << error.message;

    std::vector<box_t> rects;
    rects.reserve(array->length);

    for (int64_t i = 0; i < array->length; i++) {
      // Parse WKB
      ArrowStringView wkb_view = ArrowArrayViewGetStringUnsafe(array_view.get(), i);
      // Copy the view to a buffer because WKBReader reads from istream or byte array
      // We can cast directly if the underlying type allows
      std::stringstream iss;
      auto geom = wkb_reader.read(reinterpret_cast<const unsigned char*>(wkb_view.data),
                                  wkb_view.size_bytes);

      // Calculate Envelope for GPU Index
      const geos::geom::Envelope* env = geom->getEnvelopeInternal();

      double xmin = 0, ymin = 0, xmax = -1, ymax = -1;
      if (!env->isNull()) {
        xmin = env->getMinX();
        ymin = env->getMinY();
        xmax = env->getMaxX();
        ymax = env->getMaxY();
      }

      box_t bbox(fpoint_t((float)xmin, (float)ymin), fpoint_t((float)xmax, (float)ymax));
      rects.push_back(bbox);

      // Store User Data (global offset)
      size_t global_offset = tail_build + i;
      geom->setUserData((void*)global_offset);

      // Insert into GEOS STRtree
      tree.insert(env, geom.get());

      // Transfer ownership to storage vector
      build_geoms_storage.push_back(std::move(geom));
    }

    rt_index->PushBuild(rects.data(), rects.size());
    tail_build += array->length;

    rt_refiner->PushBuild(array_view.get());
  }

  rt_index->FinishBuilding();
  rt_refiner->FinishBuilding();

  // --- Probe Phase ---
  for (auto& probe_array : probe_arrays) {
    nanoarrow::UniqueArrayView probe_view;
    ASSERT_EQ(ArrowArrayViewInitFromSchema(probe_view.get(), probe_schema, &error),
              NANOARROW_OK)
        << error.message;
    ASSERT_EQ(ArrowArrayViewSetArray(probe_view.get(), probe_array, &error), NANOARROW_OK)
        << error.message;

    std::vector<box_t> queries;
    std::vector<std::unique_ptr<geos::geom::Geometry>> probe_geoms;
    probe_geoms.reserve(probe_array->length);

    for (int64_t i = 0; i < probe_array->length; i++) {
      ArrowBufferView wkb_view = ArrowArrayViewGetBytesUnsafe(probe_view.get(), i);
      auto geom = wkb_reader.read(wkb_view.data.as_uint8, wkb_view.size_bytes);

      const geos::geom::Envelope* env = geom->getEnvelopeInternal();

      double xmin = 0, ymin = 0, xmax = -1, ymax = -1;
      if (!env->isNull()) {
        xmin = env->getMinX();
        ymin = env->getMinY();
        xmax = env->getMaxX();
        ymax = env->getMaxY();
      }

      box_t bbox(fpoint_t((float)xmin, (float)ymin), fpoint_t((float)xmax, (float)ymax));
      queries.push_back(bbox);

      // Store user data as local offset for verification logic
      geom->setUserData((void*)i);
      probe_geoms.push_back(std::move(geom));
    }

    std::vector<uint32_t> build_indices, stream_indices;

    // GPU Probe
    rt_index->Probe(queries.data(), queries.size(), &build_indices, &stream_indices);

    // GPU Refine
    auto new_size = rt_refiner->Refine(probe_view.get(), predicate, build_indices.data(),
                                       stream_indices.data(), build_indices.size());

    build_indices.resize(new_size);
    stream_indices.resize(new_size);

    // --- CPU Verification (GEOS C++) ---
    std::vector<uint32_t> expected_build_indices;
    std::vector<uint32_t> expected_stream_indices;

    ComputeGeosJoin(build_schema, build_arrays, probe_schema,
                    std::vector<ArrowArray*>{probe_array}, predicate,
                    expected_build_indices, expected_stream_indices);

    // Assertions
    ASSERT_EQ(expected_build_indices.size(), build_indices.size());
    ASSERT_EQ(expected_stream_indices.size(), stream_indices.size());

    TestUtils::sort_vectors_by_index(expected_build_indices, expected_stream_indices);
    TestUtils::sort_vectors_by_index(build_indices, stream_indices);

    for (size_t j = 0; j < build_indices.size(); j++) {
      ASSERT_EQ(expected_build_indices[j], build_indices[j]);
      ASSERT_EQ(expected_stream_indices[j], stream_indices[j]);
    }
  }
}

TEST(JoinerTest, PIPContains) {
  using namespace TestUtils;
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  std::vector<std::string> polys{
      GetTestDataPath("synthetic_pip/polygons.parquet"),
      GetTestDataPath("countries/natural-earth_countries_geo.parquet")};
  std::vector<std::string> points{GetTestDataPath("synthetic_pip/points.parquet"),
                                  GetTestDataPath("countries/generated_points.parquet")};

  for (int i = 0; i < polys.size(); i++) {
    auto poly_path = TestUtils::GetTestDataPath(polys[i]);
    auto point_path = TestUtils::GetCanonicalPath(points[i]);
    auto poly_arrays = ReadParquet(poly_path, 1000);
    auto point_arrays = ReadParquet(point_path, 1000);
    std::vector<nanoarrow::UniqueArray> poly_uniq_arrays, point_uniq_arrays;
    std::vector<nanoarrow::UniqueSchema> poly_uniq_schema, point_uniq_schema;

    for (auto& arr : poly_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, poly_uniq_arrays.emplace_back().get(),
                                            poly_uniq_schema.emplace_back().get()));
    }
    for (auto& arr : point_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, point_uniq_arrays.emplace_back().get(),
                                            point_uniq_schema.emplace_back().get()));
    }

    std::vector<ArrowArray*> poly_c_arrays, point_c_arrays;
    for (auto& arr : poly_uniq_arrays) {
      poly_c_arrays.push_back(arr.get());
    }
    for (auto& arr : point_uniq_arrays) {
      point_c_arrays.push_back(arr.get());
    }
    TestJoiner(poly_uniq_schema[0].get(), poly_c_arrays, point_uniq_schema[0].get(),
               point_c_arrays, Predicate::kContains);
  }
}

TEST(JoinerTest, PIPContainsPipelined) {
  using namespace TestUtils;
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  std::vector<std::string> polys{
      GetTestDataPath("synthetic_pip/polygons.parquet"),
      GetTestDataPath("countries/natural-earth_countries_geo.parquet")};
  std::vector<std::string> points{GetTestDataPath("synthetic_pip/points.parquet"),
                                  GetTestDataPath("countries/generated_points.parquet")};

  for (int i = 0; i < polys.size(); i++) {
    auto poly_path = TestUtils::GetTestDataPath(polys[i]);
    auto point_path = TestUtils::GetCanonicalPath(points[i]);
    auto poly_arrays = ReadParquet(poly_path, 1000);
    auto point_arrays = ReadParquet(point_path, 1000);
    std::vector<nanoarrow::UniqueArray> poly_uniq_arrays, point_uniq_arrays;
    std::vector<nanoarrow::UniqueSchema> poly_uniq_schema, point_uniq_schema;

    for (auto& arr : poly_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, poly_uniq_arrays.emplace_back().get(),
                                            poly_uniq_schema.emplace_back().get()));
    }
    for (auto& arr : point_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, point_uniq_arrays.emplace_back().get(),
                                            point_uniq_schema.emplace_back().get()));
    }

    std::vector<ArrowArray*> poly_c_arrays, point_c_arrays;
    for (auto& arr : poly_uniq_arrays) {
      poly_c_arrays.push_back(arr.get());
    }
    for (auto& arr : point_uniq_arrays) {
      point_c_arrays.push_back(arr.get());
    }
    TestJoiner(poly_uniq_schema[0].get(), poly_c_arrays, point_uniq_schema[0].get(),
               point_c_arrays, Predicate::kContains, true);
  }
}

TEST(JoinerTest, PIPWithin) {
  using namespace TestUtils;
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  std::vector<std::string> polys{
      GetTestDataPath("synthetic_pip/polygons.parquet"),
      GetTestDataPath("countries/natural-earth_countries_geo.parquet")};
  std::vector<std::string> points{GetTestDataPath("synthetic_pip/points.parquet"),
                                  GetTestDataPath("countries/generated_points.parquet")};

  for (int i = 0; i < polys.size(); i++) {
    auto poly_path = TestUtils::GetTestDataPath(polys[i]);
    auto point_path = TestUtils::GetCanonicalPath(points[i]);
    auto poly_arrays = ReadParquet(poly_path, 1000);
    auto point_arrays = ReadParquet(point_path, 1000);
    std::vector<nanoarrow::UniqueArray> poly_uniq_arrays, point_uniq_arrays;
    std::vector<nanoarrow::UniqueSchema> poly_uniq_schema, point_uniq_schema;

    for (auto& arr : poly_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, poly_uniq_arrays.emplace_back().get(),
                                            poly_uniq_schema.emplace_back().get()));
    }
    for (auto& arr : point_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, point_uniq_arrays.emplace_back().get(),
                                            point_uniq_schema.emplace_back().get()));
    }

    std::vector<ArrowArray*> poly_c_arrays, point_c_arrays;
    for (auto& arr : poly_uniq_arrays) {
      poly_c_arrays.push_back(arr.get());
    }
    for (auto& arr : point_uniq_arrays) {
      point_c_arrays.push_back(arr.get());
    }
    TestJoiner(point_uniq_schema[0].get(), point_c_arrays, poly_uniq_schema[0].get(),
               poly_c_arrays, Predicate::kWithin);
  }
}

TEST(JoinerTest, PolygonPolygonIntersects) {
  using namespace TestUtils;
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  std::vector<std::string> polys1{GetTestDataPath("synthetic_poly/polygons1.parquet")};
  std::vector<std::string> polys2{GetTestDataPath("synthetic_poly/polygons2.parquet")};

  for (int i = 0; i < polys1.size(); i++) {
    auto poly1_path = TestUtils::GetTestDataPath(polys1[i]);
    auto poly2_path = TestUtils::GetCanonicalPath(polys2[i]);
    auto poly1_arrays = ReadParquet(poly1_path, 1000);
    auto point2_arrays = ReadParquet(poly2_path, 1000);
    std::vector<nanoarrow::UniqueArray> poly1_uniq_arrays, poly2_uniq_arrays;
    std::vector<nanoarrow::UniqueSchema> poly1_uniq_schema, poly2_uniq_schema;

    for (auto& arr : poly1_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, poly1_uniq_arrays.emplace_back().get(),
                                            poly1_uniq_schema.emplace_back().get()));
    }
    for (auto& arr : point2_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, poly2_uniq_arrays.emplace_back().get(),
                                            poly2_uniq_schema.emplace_back().get()));
    }

    std::vector<ArrowArray*> poly1_c_arrays, poly2_c_arrays;
    for (auto& arr : poly1_uniq_arrays) {
      poly1_c_arrays.push_back(arr.get());
    }
    for (auto& arr : poly2_uniq_arrays) {
      poly2_c_arrays.push_back(arr.get());
    }
    TestJoiner(poly1_uniq_schema[0].get(), poly1_c_arrays, poly2_uniq_schema[0].get(),
               poly2_c_arrays, Predicate::kIntersects);
  }
}

void TestJoinerWKT(const std::vector<std::vector<std::string>>& build_wkt_batches,
                   const std::vector<std::vector<std::string>>& probe_wkt_batches,
                   Predicate predicate, bool pipelined = false) {
  nanoarrow::UniqueArrayStream build_stream, probe_stream;
  ArrayStreamFromWKT(build_wkt_batches, GEOARROW_TYPE_WKB, build_stream.get());
  ArrayStreamFromWKT(probe_wkt_batches, GEOARROW_TYPE_WKB, probe_stream.get());

  nanoarrow::UniqueSchema build_schema, probe_schema;
  ArrowError error;
  ASSERT_EQ(ArrowArrayStreamGetSchema(build_stream.get(), build_schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayStreamGetSchema(probe_stream.get(), probe_schema.get(), &error),
            NANOARROW_OK)
      << error.message;

  std::vector<nanoarrow::UniqueArray> build_uniq_arrays, probe_uniq_arrays;
  while (true) {
    nanoarrow::UniqueArray arr;
    ASSERT_EQ(ArrowArrayStreamGetNext(build_stream.get(), arr.get(), &error),
              NANOARROW_OK)
        << error.message;
    if (arr->release == nullptr) break;
    build_uniq_arrays.push_back(std::move(arr));
  }
  while (true) {
    nanoarrow::UniqueArray arr;
    ASSERT_EQ(ArrowArrayStreamGetNext(probe_stream.get(), arr.get(), &error),
              NANOARROW_OK)
        << error.message;
    if (arr->release == nullptr) break;
    probe_uniq_arrays.push_back(std::move(arr));
  }

  std::vector<ArrowArray*> build_c_arrays, probe_c_arrays;
  for (auto& arr : build_uniq_arrays) {
    build_c_arrays.push_back(arr.get());
  }
  for (auto& arr : probe_uniq_arrays) {
    probe_c_arrays.push_back(arr.get());
  }

  TestJoiner(build_schema.get(), build_c_arrays, probe_schema.get(), probe_c_arrays,
             predicate, pipelined);
}

TEST(JoinerTest, GeometryCollectionChildPolygonInPass2) {
  std::vector<std::vector<std::string>> build_batches = {{
      "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
  }};
  std::vector<std::vector<std::string>> probe_batches = {{
      "GEOMETRYCOLLECTION (POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1)), POINT (100 100))",
  }};
  TestJoinerWKT(build_batches, probe_batches, Predicate::kIntersects);
  TestJoinerWKT(probe_batches, build_batches, Predicate::kIntersects);
}

TEST(JoinerTest, GeosGeometryCollectionTestSuite) {
  struct TestCase {
    std::string name;
    std::string wkt_a;
    std::string wkt_b;
  };

  // Test cases adapted from GEOS RelateNGGC test suite and GeometryCollection edge cases
  std::vector<TestCase> cases = {
      {"testMP_GLP_GEOS1027",
       "MULTIPOLYGON (((0 0, 3 0, 3 3, 0 3, 0 0)))",
       "GEOMETRYCOLLECTION (LINESTRING (1 2, 1 1), POINT (0 0))"},
      {"testGPL_A",
       "GEOMETRYCOLLECTION (POINT (7 1), LINESTRING (6 5, 6 4))",
       "POLYGON ((7 1, 1 3, 3 9, 7 1))"},
      {"testP_GPL",
       "POINT (0 0)",
       "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 0))"},
      {"testLineInOverlappingPolygonsTouchingInteriorEdge",
       "LINESTRING (3 7, 7 3)",
       "GEOMETRYCOLLECTION (POLYGON ((1 9, 7 9, 7 3, 1 3, 1 9)), POLYGON ((9 1, 3 1, 3 7, 9 7, 9 1)))"},
      {"testLineInOverlappingPolygonsCrossingInteriorEdgeAtVertex",
       "LINESTRING (2 2, 8 8)",
       "GEOMETRYCOLLECTION (POLYGON ((1 1, 1 7, 7 7, 7 1, 1 1)), POLYGON ((9 9, 9 3, 3 3, 3 9, 9 9)))"},
      {"testLineInOverlappingPolygonsCrossingInteriorEdgeProper",
       "LINESTRING (2 4, 6 8)",
       "GEOMETRYCOLLECTION (POLYGON ((1 1, 1 7, 7 7, 7 1, 1 1)), POLYGON ((9 9, 9 3, 3 3, 3 9, 9 9)))"},
      {"testPolygonInOverlappingPolygonsTouchingBoundaries",
       "GEOMETRYCOLLECTION (POLYGON ((1 9, 6 9, 6 4, 1 4, 1 9)), POLYGON ((9 1, 4 1, 4 6, 9 6, 9 1)))",
       "POLYGON ((2 6, 6 2, 8 4, 4 8, 2 6))"},
      {"testLineInOverlappingPolygonsBoundaries",
       "LINESTRING (1 6, 9 6, 9 1, 1 1, 1 6)",
       "GEOMETRYCOLLECTION (POLYGON ((1 1, 1 6, 6 6, 6 1, 1 1)), POLYGON ((9 1, 4 1, 4 6, 9 6, 9 1)))"},
      {"testAdjacentPolygonsContainedInAdjacentPolygons",
       "GEOMETRYCOLLECTION (POLYGON ((2 2, 2 5, 4 5, 4 2, 2 2)), POLYGON ((8 2, 4 3, 4 4, 8 5, 8 2)))",
       "GEOMETRYCOLLECTION (POLYGON ((1 1, 1 6, 4 6, 4 1, 1 1)), POLYGON ((9 1, 4 1, 4 6, 9 6, 9 1)))"},
      {"testGCMultiPolygonIntersectsPolygon",
       "POLYGON ((2 5, 3 5, 3 3, 2 3, 2 5))",
       "GEOMETRYCOLLECTION (MULTIPOLYGON (((1 4, 4 4, 4 1, 1 1, 1 4)), ((5 4, 8 4, 8 1, 5 1, 5 4))))"},
      {"testPolygonContainsGCMultiPolygonElement",
       "POLYGON ((0 5, 4 5, 4 1, 0 1, 0 5))",
       "GEOMETRYCOLLECTION (MULTIPOLYGON (((1 4, 3 4, 3 2, 1 2, 1 4)), ((6 4, 8 4, 8 2, 6 2, 6 4))))"},
      {"testPolygonOverlappingGCPolygon",
       "GEOMETRYCOLLECTION (POLYGON ((18.6 40.8, 16.8825 39.618567, 16.9319 39.5461, 17.10985 39.485133, 16.6143 38.4302, 16.43145 38.313267, 16.2 37.5, 14.8 37.8, 14.96475 40.474933, 18.6 40.8)))",
       "POLYGON ((16.3649953125 38.37219358064516, 16.3649953125 39.545924774193544, 17.949465625000002 39.545924774193544, 17.949465625000002 38.37219358064516, 16.3649953125 38.37219358064516))"},
      {"testAdjPolygonsCoverPolygonWithEndpointInside",
       "GEOMETRYCOLLECTION (POLYGON ((5 5, 2 9, 9 9, 9 5, 5 5)), POLYGON ((3 1, 5 5, 9 5, 9 1, 3 1)), POLYGON ((1 9, 2 9, 5 5, 3 1, 1 1, 1 9)))",
       "POLYGON ((3 7, 7 7, 7 3, 3 3, 3 7))"},
      {"testAdjPolygonsCoverPointAtNode",
       "GEOMETRYCOLLECTION (POLYGON ((5 5, 2 9, 9 9, 9 5, 5 5)), POLYGON ((3 1, 5 5, 9 5, 9 1, 3 1)), POLYGON ((1 9, 2 9, 5 5, 3 1, 1 1, 1 9)))",
       "POINT (5 5)"},
      {"testAdjPolygonsCoverPointOnEdge",
       "GEOMETRYCOLLECTION (POLYGON ((5 5, 2 9, 9 9, 9 5, 5 5)), POLYGON ((3 1, 5 5, 9 5, 9 1, 3 1)), POLYGON ((1 9, 2 9, 5 5, 3 1, 1 1, 1 9)))",
       "POINT (7 5)"},
      {"testAdjPolygonsContainingPolygonTouchingInteriorEndpoint",
       "GEOMETRYCOLLECTION (POLYGON ((5 5, 2 9, 9 9, 9 5, 5 5)), POLYGON ((3 1, 5 5, 9 5, 9 1, 3 1)), POLYGON ((1 9, 2 9, 5 5, 3 1, 1 1, 1 9)))",
       "POLYGON ((5 5, 7 5, 7 3, 5 3, 5 5))"},
      {"testAdjPolygonsOverlappedByPolygonWithHole",
       "GEOMETRYCOLLECTION (POLYGON ((5 5, 2 9, 9 9, 9 5, 5 5)), POLYGON ((3 1, 5 5, 9 5, 9 1, 3 1)), POLYGON ((1 9, 2 9, 5 5, 3 1, 1 1, 1 9)))",
       "POLYGON ((0 10, 10 10, 10 0, 0 0, 0 10), (2 8, 8 8, 8 2, 2 2, 2 8))"},
      {"testAdjPolygonsContainingLine",
       "GEOMETRYCOLLECTION (POLYGON ((5 5, 2 9, 9 9, 9 5, 5 5)), POLYGON ((3 1, 5 5, 9 5, 9 1, 3 1)), POLYGON ((1 9, 2 9, 5 5, 3 1, 1 1, 1 9)))",
       "LINESTRING (5 5, 7 7)"},
      {"testAdjPolygonsContainingLineAndPoint",
       "GEOMETRYCOLLECTION (POLYGON ((5 5, 2 9, 9 9, 9 5, 5 5)), POLYGON ((3 1, 5 5, 9 5, 9 1, 3 1)), POLYGON ((1 9, 2 9, 5 5, 3 1, 1 1, 1 9)))",
       "GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (5 7, 7 7))"},
      {"testPolygonContainingPointsInBoundary",
       "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))",
       "GEOMETRYCOLLECTION (POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0)), MULTIPOINT ((0 2), (0 5)))"},
      {"testPolygonContainingLineInBoundary",
       "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))",
       "GEOMETRYCOLLECTION (POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0)), LINESTRING (0 2, 0 5))"},
      {"testPolygonContainingBentLineInBoundary",
       "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))",
       "GEOMETRYCOLLECTION (POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0)), LINESTRING (0 2, 0 5, 5 5))"},
      {"testGCContainedPolygonWithDistantPoint",
       "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
       "GEOMETRYCOLLECTION (POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1)), POINT (100 100))"},
      {"testGCContainedMultiPolygonWithDistantPoint",
       "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
       "GEOMETRYCOLLECTION (MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((3 3, 4 3, 4 4, 3 4, 3 3))), POINT (100 100))"},
      {"testGCContainedLineWithDistantPoint",
       "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
       "GEOMETRYCOLLECTION (LINESTRING (1 1, 2 2), POINT (100 100))"},
      {"testTwoGCsWithDistantNonIntersectingAndContainedComponents",
       "GEOMETRYCOLLECTION (POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0)), POINT (-100 -100))",
       "GEOMETRYCOLLECTION (POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1)), POINT (100 100))"},
  };

  for (const auto& tc : cases) {
    SCOPED_TRACE(tc.name);
    TestJoinerWKT({{tc.wkt_a}}, {{tc.wkt_b}}, Predicate::kIntersects);
    TestJoinerWKT({{tc.wkt_b}}, {{tc.wkt_a}}, Predicate::kIntersects);
  }
}

TEST(JoinerTest, MixedGeometryProbeIntersects) {
  // Build side has concave / L-shaped polygons and multipolygons where MBR != geometry
  std::vector<std::vector<std::string>> build_batches = {{
      // L-shaped polygon: MBR is [0, 0, 10, 10], but upper-right [2, 2, 10, 10] is empty!
      "POLYGON ((0 0, 10 0, 10 2, 2 2, 2 10, 0 10, 0 0))",
      // Polygon with hole: MBR is [20, 20, 30, 30], hole is [22, 22, 28, 28]
      "POLYGON ((20 20, 30 20, 30 30, 20 30, 20 20), (22 22, 28 22, 28 28, 22 28, 22 22))",
      // MultiPolygon with two disjoint parts: MBR is [40, 40, 60, 60], gap in middle
      "MULTIPOLYGON (((40 40, 45 40, 45 45, 40 45, 40 40)), ((55 55, 60 55, 60 60, 55 60, 55 55)))",
  }};

  // Probe side mixes Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon
  // causing ParallelWkbLoader to upcast the probe column to GeometryCollection.
  std::vector<std::vector<std::string>> probe_batches = {{
      // Points inside MBR but outside actual polygon (false positive if only MBR-checked)
      "POINT (5 5)",
      "POINT (8 8)",
      "POINT (25 25)",
      "POINT (50 50)",
      // Points truly inside polygon
      "POINT (1 1)",
      "POINT (21 21)",
      "POINT (42 42)",
      "POINT (57 57)",
      // MultiPoints: one outside (in MBR gap), one partially inside, one fully inside
      "MULTIPOINT ((5 5), (8 8))",
      "MULTIPOINT ((5 5), (1 1))",
      "MULTIPOINT ((1 1), (1 5))",
      // LineStrings: one inside MBR gap, one crossing polygon
      "LINESTRING (4 4, 9 9)",
      "LINESTRING (0 5, 5 5)",
      // MultiLineStrings: one in hole, one crossing hole boundary
      "MULTILINESTRING ((23 23, 27 27), (24 24, 26 26))",
      "MULTILINESTRING ((23 23, 27 27), (19 25, 21 25))",
      // Polygons: one inside L-shape gap, one overlapping L-shape, one properly contained
      "POLYGON ((4 4, 9 4, 9 9, 4 9, 4 4))",
      "POLYGON ((0.5 0.5, 5 0.5, 5 5, 0.5 5, 0.5 0.5))",
      "POLYGON ((0.2 0.2, 1.8 0.2, 1.8 1.8, 0.2 1.8, 0.2 0.2))",
      // MultiPolygons: one inside MultiPolygon gap, one overlapping
      "MULTIPOLYGON (((47 47, 52 47, 52 52, 47 52, 47 47)))",
      "MULTIPOLYGON (((44 44, 48 44, 48 48, 44 48, 44 44)))",
  }};

  TestJoinerWKT(build_batches, probe_batches, Predicate::kIntersects);
  TestJoinerWKT(build_batches, probe_batches, Predicate::kContains);

  // Also test ST_Intersects with nested multi-part GeometryCollections
  auto probe_with_nested_gc = probe_batches;
  probe_with_nested_gc[0].push_back("GEOMETRYCOLLECTION (POINT (5 5), POINT (1 1))");
  probe_with_nested_gc[0].push_back("GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (4 4, 9 9))");
  TestJoinerWKT(build_batches, probe_with_nested_gc, Predicate::kIntersects);
}

TEST(JoinerTest, MixedGeometryBuildAndBothSidesIntersects) {
  // Contains all 6 geometry types (Point, MultiPoint, LineString, MultiLineString,
  // Polygon, MultiPolygon) so all 6 sub-array branches in EvaluateGeometryCollection
  // are exercised on both build and probe sides.
  std::vector<std::vector<std::string>> mixed_all_6_types_1 = {{
      // 1. Point (inside L-shape gap vs inside L-shape)
      "POINT (5 5)",
      "POINT (1 1)",
      // 2. MultiPoint (in gap vs partially inside L-shape)
      "MULTIPOINT ((5 5), (8 8))",
      "MULTIPOINT ((5 5), (1 1))",
      // 3. LineString (in gap vs crossing L-shape)
      "LINESTRING (4 4, 9 9)",
      "LINESTRING (0 5, 5 5)",
      // 4. MultiLineString (in hole vs crossing hole boundary)
      "MULTILINESTRING ((23 23, 27 27), (24 24, 26 26))",
      "MULTILINESTRING ((23 23, 27 27), (19 25, 21 25))",
      // 5. Polygon (L-shaped polygon and polygon with hole)
      "POLYGON ((0 0, 10 0, 10 2, 2 2, 2 10, 0 10, 0 0))",
      "POLYGON ((20 20, 30 20, 30 30, 20 30, 20 20), (22 22, 28 22, 28 28, 22 28, 22 22))",
      // 6. MultiPolygon (two disjoint parts with gap in [45, 55])
      "MULTIPOLYGON (((40 40, 45 40, 45 45, 40 45, 40 40)), ((55 55, 60 55, 60 60, 55 60, 55 55)))",
  }};

  std::vector<std::vector<std::string>> mixed_all_6_types_2 = {{
      // 1. Point
      "POINT (5 5)",
      "POINT (1 1)",
      "POINT (25 25)",
      "POINT (21 21)",
      "POINT (50 50)",
      "POINT (42 42)",
      // 2. MultiPoint
      "MULTIPOINT ((5 5), (8 8))",
      "MULTIPOINT ((1 1), (5 5))",
      "MULTIPOINT ((42 42), (50 50))",
      // 3. LineString
      "LINESTRING (4 4, 9 9)",
      "LINESTRING (0 5, 5 5)",
      "LINESTRING (46 46, 54 54)",
      // 4. MultiLineString
      "MULTILINESTRING ((4 4, 9 9), (6 6, 8 8))",
      "MULTILINESTRING ((0 1, 5 1), (46 46, 54 54))",
      // 5. Polygon
      "POLYGON ((4 4, 9 4, 9 9, 4 9, 4 4))",
      "POLYGON ((0.5 0.5, 5 0.5, 5 5, 0.5 5, 0.5 0.5))",
      // 6. MultiPolygon
      "MULTIPOLYGON (((47 47, 52 47, 52 52, 47 52, 47 47)))",
      "MULTIPOLYGON (((44 44, 48 44, 48 48, 44 48, 44 44)))",
  }};

  // Homogeneous batches for each of the 6 geometry types
  std::vector<std::vector<std::string>> points_only = {{
      "POINT (5 5)", "POINT (1 1)", "POINT (21 21)", "POINT (25 25)", "POINT (42 42)", "POINT (50 50)",
  }};
  std::vector<std::vector<std::string>> multipoints_only = {{
      "MULTIPOINT ((5 5), (8 8))", "MULTIPOINT ((1 1), (5 5))", "MULTIPOINT ((42 42), (50 50))",
  }};
  std::vector<std::vector<std::string>> linestrings_only = {{
      "LINESTRING (4 4, 9 9)", "LINESTRING (0 5, 5 5)", "LINESTRING (23 23, 27 27)",
  }};
  std::vector<std::vector<std::string>> multilinestrings_only = {{
      "MULTILINESTRING ((4 4, 9 9), (6 6, 8 8))", "MULTILINESTRING ((0 5, 5 5), (20 20, 21 21))",
  }};
  std::vector<std::vector<std::string>> polygons_only = {{
      "POLYGON ((4 4, 9 4, 9 9, 4 9, 4 4))",
      "POLYGON ((0.5 0.5, 5 0.5, 5 5, 0.5 5, 0.5 0.5))",
      "POLYGON ((23 23, 27 23, 27 27, 23 27, 23 23))",
  }};
  std::vector<std::vector<std::string>> multipolygons_only = {{
      "MULTIPOLYGON (((47 47, 52 47, 52 52, 47 52, 47 47)))",
      "MULTIPOLYGON (((44 44, 48 44, 48 48, 44 48, 44 44)))",
  }};

  // 1. Build side is GeometryCollection (all 6 types, single-geom per row) vs each homogeneous probe type
  for (const auto& probe_batch : {points_only, multipoints_only, linestrings_only,
                                  multilinestrings_only, polygons_only, multipolygons_only}) {
    TestJoinerWKT(mixed_all_6_types_1, probe_batch, Predicate::kIntersects);
    TestJoinerWKT(probe_batch, mixed_all_6_types_1, Predicate::kIntersects);
  }
  for (const auto& probe_batch :
       {points_only, multipoints_only, linestrings_only, multilinestrings_only}) {
    TestJoinerWKT(mixed_all_6_types_1, probe_batch, Predicate::kContains);
    TestJoinerWKT(probe_batch, mixed_all_6_types_1, Predicate::kWithin);
  }

  // 2. Both build and probe sides are GeometryCollection (all 6 types, single-geom fast path)
  TestJoinerWKT(mixed_all_6_types_1, mixed_all_6_types_2, Predicate::kIntersects);

  // 3. Multi-element GEOMETRYCOLLECTIONs covering all 6 child geometry types + POINT EMPTY + GC EMPTY
  // (exercises the general expansion + deduplication path in EvaluateGeometryCollection)
  auto multi_gc_side_1 = mixed_all_6_types_1;
  multi_gc_side_1[0].push_back("GEOMETRYCOLLECTION EMPTY");
  multi_gc_side_1[0].push_back("GEOMETRYCOLLECTION (POINT EMPTY, POINT (1 1))");
  multi_gc_side_1[0].push_back("GEOMETRYCOLLECTION (POINT EMPTY, POINT (5 5))");
  multi_gc_side_1[0].push_back(
      "GEOMETRYCOLLECTION ("
      "POINT (1 1), "
      "MULTIPOINT ((5 5), (1 1)), "
      "LINESTRING (0 5, 5 5), "
      "MULTILINESTRING ((23 23, 27 27), (19 25, 21 25)), "
      "POLYGON ((0 0, 10 0, 10 2, 2 2, 2 10, 0 10, 0 0)), "
      "MULTIPOLYGON (((40 40, 45 40, 45 45, 40 45, 40 40)), ((55 55, 60 55, 60 60, 55 60, 55 55)))"
      ")");

  auto multi_gc_side_2 = mixed_all_6_types_2;
  multi_gc_side_2[0].push_back("GEOMETRYCOLLECTION EMPTY");
  multi_gc_side_2[0].push_back("GEOMETRYCOLLECTION (POINT EMPTY, POINT (5 5))");
  multi_gc_side_2[0].push_back(
      "GEOMETRYCOLLECTION ("
      "POINT (5 5), "
      "MULTIPOINT ((5 5), (8 8)), "
      "LINESTRING (4 4, 9 9), "
      "MULTILINESTRING ((4 4, 9 9), (6 6, 8 8)), "
      "POLYGON ((4 4, 9 4, 9 9, 4 9, 4 4)), "
      "MULTIPOLYGON (((47 47, 52 47, 52 52, 47 52, 47 47)))"
      ")");

  TestJoinerWKT(multi_gc_side_1, multi_gc_side_2, Predicate::kIntersects);
}

namespace {

bool EvaluatePreparedGeosPredicate(
    Predicate predicate, const geos::geom::prep::PreparedGeometry* prep_build,
    const geos::geom::Geometry* build_geom, const geos::geom::Geometry* probe_geom) {
  switch (predicate) {
    case Predicate::kIntersects:
      return prep_build->intersects(probe_geom);
    case Predicate::kContains:
      return prep_build->contains(probe_geom);
    case Predicate::kCovers:
      return prep_build->covers(probe_geom);
    case Predicate::kWithin:
      return prep_build->within(probe_geom);
    case Predicate::kCoveredBy:
      return prep_build->coveredBy(probe_geom);
    case Predicate::kTouches:
      return prep_build->touches(probe_geom);
    case Predicate::kDisjoint:
      return prep_build->disjoint(probe_geom);
    case Predicate::kEquals:
      return build_geom->equals(probe_geom);
    default:
      throw std::out_of_range("Unsupported predicate");
  }
}

std::string ResolveSyntheticPolyPath(const std::string& rel_file) {
  const char* env_dir = std::getenv("GPUSPATIAL_TEST_DIR");
  if (env_dir != nullptr && std::string_view(env_dir) != "") {
    return (std::filesystem::path(env_dir) / rel_file).string();
  }
  std::filesystem::path fallback =
      std::filesystem::path(__FILE__).parent_path() / "data" / rel_file;
  if (std::filesystem::exists(fallback)) {
    return fallback.string();
  }
  return TestUtils::GetTestDataPath(rel_file);
}

}  // namespace

void RunPolygonBenchmarkSuite(const std::string& suite_name, ArrowSchema* build_schema,
                              ArrowArray* build_array, ArrowSchema* probe_schema,
                              ArrowArray* probe_array) {
  using namespace TestUtils;
  using coord_t = double;
  using fpoint_t = Point<coord_t, 2>;
  using box_t = Box<fpoint_t>;
  using Clock = std::chrono::high_resolution_clock;

  ArrowError error;
  nanoarrow::UniqueArrayView build_view, probe_view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(build_view.get(), build_schema, &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(build_view.get(), build_array, &error), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewInitFromSchema(probe_view.get(), probe_schema, &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(probe_view.get(), probe_array, &error), NANOARROW_OK);

  int64_t n_build = build_array->length;
  int64_t n_probe = probe_array->length;

  std::vector<ArrowBufferView> build_wkbs(n_build), probe_wkbs(n_probe);
  std::vector<box_t> build_boxes(n_build), probe_boxes(n_probe);

  auto geos_factory = geos::geom::GeometryFactory::create();
  geos::io::WKBReader wkb_reader(*geos_factory);

  for (int64_t i = 0; i < n_build; ++i) {
    build_wkbs[i] = ArrowArrayViewGetBytesUnsafe(build_view.get(), i);
    auto g = wkb_reader.read(build_wkbs[i].data.as_uint8, build_wkbs[i].size_bytes);
    const auto* env = g->getEnvelopeInternal();
    build_boxes[i] = box_t(fpoint_t(env->getMinX(), env->getMinY()),
                           fpoint_t(env->getMaxX(), env->getMaxY()));
  }
  for (int64_t i = 0; i < n_probe; ++i) {
    probe_wkbs[i] = ArrowArrayViewGetBytesUnsafe(probe_view.get(), i);
    auto g = wkb_reader.read(probe_wkbs[i].data.as_uint8, probe_wkbs[i].size_bytes);
    const auto* env = g->getEnvelopeInternal();
    probe_boxes[i] = box_t(fpoint_t(env->getMinX(), env->getMinY()),
                           fpoint_t(env->getMaxX(), env->getMaxY()));
  }

  auto rt_engine = std::make_shared<RTEngine>();
  rt_engine->Init(get_default_rt_config(TestUtils::GetTestShaderPath()));

  RTSpatialIndexConfig idx_config;
  idx_config.rt_engine = rt_engine;
  auto rt_index = CreateRTSpatialIndex<coord_t, 2>(idx_config);
  rt_index->PushBuild(build_boxes.data(), build_boxes.size());
  rt_index->FinishBuilding();

  std::vector<uint32_t> cand_build_indices, cand_probe_indices;
  rt_index->Probe(probe_boxes.data(), probe_boxes.size(), &cand_build_indices,
                  &cand_probe_indices);
  size_t num_candidates = cand_build_indices.size();

  unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());

  // 1. Multi-threaded GEOS WKB parse + PreparedPolygon build
  std::vector<std::unique_ptr<geos::geom::Geometry>> geos_build_geoms(n_build);
  std::vector<std::unique_ptr<geos::geom::prep::PreparedGeometry>> geos_prep_build(
      n_build);
  std::vector<std::unique_ptr<geos::geom::Geometry>> geos_probe_geoms(n_probe);

  auto t_geos_build_start = Clock::now();
  {
    std::vector<std::thread> workers;
    for (unsigned int tid = 0; tid < num_threads; ++tid) {
      workers.emplace_back([&, tid]() {
        auto local_factory = geos::geom::GeometryFactory::create();
        geos::io::WKBReader local_reader(*local_factory);
        for (int64_t i = tid; i < n_build; i += num_threads) {
          geos_build_geoms[i] =
              local_reader.read(build_wkbs[i].data.as_uint8, build_wkbs[i].size_bytes);
          geos_build_geoms[i]->getEnvelopeInternal();
          geos_prep_build[i] =
              geos::geom::prep::PreparedGeometryFactory::prepare(geos_build_geoms[i].get());
        }
        for (int64_t i = tid; i < n_probe; i += num_threads) {
          geos_probe_geoms[i] =
              local_reader.read(probe_wkbs[i].data.as_uint8, probe_wkbs[i].size_bytes);
          geos_probe_geoms[i]->getEnvelopeInternal();
        }
      });
    }
    for (auto& w : workers) w.join();
  }
  double geos_build_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - t_geos_build_start)
          .count();

  // 2. GPU ParallelWkbLoader + RelateEngine & RTSpatialRefiner setup
  RTSpatialRefinerConfig refiner_config;
  refiner_config.rt_engine = rt_engine;
  auto rt_refiner = CreateRTSpatialRefiner(refiner_config);

  auto cuda_stream = rmm::cuda_stream_default;
  ParallelWkbLoader<Point<double, 2>, uint32_t> build_loader, probe_loader;
  build_loader.Init();
  probe_loader.Init();

  auto t_gpu_build_start = Clock::now();
  build_loader.Parse(cuda_stream, build_view.get(), 0, n_build);
  auto d_build_geoms = build_loader.Finish(cuda_stream);
  probe_loader.Parse(cuda_stream, probe_view.get(), 0, n_probe);
  auto d_probe_geoms = probe_loader.Finish(cuda_stream);
  cuda_stream.synchronize();
  double gpu_build_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - t_gpu_build_start).count();

  rt_refiner->PushBuild(build_view.get());
  rt_refiner->FinishBuilding();

  RelateEngine<Point<double, 2>, uint32_t> relate_engine(&d_build_geoms, rt_engine.get());

  std::cout << "\n====================================================================\n"
            << "[" << suite_name << "] Polygon-Polygon Benchmark & Parity (Threads = "
            << num_threads << ")\n"
            << "Build polygons: " << n_build << ", Probe polygons: " << n_probe
            << ", Candidate pairs: " << num_candidates << "\n"
            << "WKB Parse + Index Prep -> GPU: " << std::fixed << std::setprecision(2)
            << gpu_build_ms
            << " ms | GEOS PreparedPolygon (MT): " << geos_build_ms << " ms\n"
            << "--------------------------------------------------------------------\n";

  const std::vector<Predicate> predicates = {
      Predicate::kIntersects, Predicate::kContains,  Predicate::kCovers,
      Predicate::kWithin,     Predicate::kCoveredBy, Predicate::kTouches};

  for (Predicate pred : predicates) {
    // A. Pure GPU RT RelateEngine::Evaluate (pre-loaded DeviceGeometries, apples-to-apples)
    rmm::device_uvector<uint32_t> d_b(num_candidates, cuda_stream);
    rmm::device_uvector<uint32_t> d_p(num_candidates, cuda_stream);
    CUDA_CHECK(cudaMemcpyAsync(d_b.data(), cand_build_indices.data(),
                               num_candidates * sizeof(uint32_t), cudaMemcpyHostToDevice,
                               cuda_stream.value()));
    CUDA_CHECK(cudaMemcpyAsync(d_p.data(), cand_probe_indices.data(),
                               num_candidates * sizeof(uint32_t), cudaMemcpyHostToDevice,
                               cuda_stream.value()));
    cuda_stream.synchronize();

    auto t_gpu_eval_start = Clock::now();
    relate_engine.Evaluate(cuda_stream, d_probe_geoms, pred, d_b, d_p);
    cuda_stream.synchronize();
    double gpu_eval_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t_gpu_eval_start)
            .count();

    // B. End-to-End RTSpatialRefiner::Refine (includes probe WKB parsing)
    std::vector<uint32_t> gpu_b = cand_build_indices;
    std::vector<uint32_t> gpu_p = cand_probe_indices;
    auto t_gpu_ref_start = Clock::now();
    size_t gpu_count = rt_refiner->Refine(probe_view.get(), pred, gpu_b.data(),
                                          gpu_p.data(), gpu_b.size());
    double gpu_refine_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t_gpu_ref_start).count();
    gpu_b.resize(gpu_count);
    gpu_p.resize(gpu_count);

    // C. Multi-threaded GEOS PreparedPolygon Refine
    std::vector<uint8_t> geos_keep(num_candidates, 0);
    auto t_geos_ref_start = Clock::now();
    {
      std::vector<std::thread> workers;
      for (unsigned int tid = 0; tid < num_threads; ++tid) {
        workers.emplace_back([&, tid]() {
          for (size_t k = 0; k < num_candidates; ++k) {
            uint32_t b_id = cand_build_indices[k];
            if (b_id % num_threads != tid) continue;
            uint32_t p_id = cand_probe_indices[k];
            if (EvaluatePreparedGeosPredicate(pred, geos_prep_build[b_id].get(),
                                              geos_build_geoms[b_id].get(),
                                              geos_probe_geoms[p_id].get())) {
              geos_keep[k] = 1;
            }
          }
        });
      }
      for (auto& w : workers) w.join();
    }
    double geos_refine_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t_geos_ref_start)
            .count();

    std::vector<std::pair<uint32_t, uint32_t>> gpu_pairs, geos_pairs;
    for (size_t k = 0; k < gpu_count; ++k) {
      gpu_pairs.emplace_back(gpu_b[k], gpu_p[k]);
    }
    for (size_t k = 0; k < num_candidates; ++k) {
      if (geos_keep[k])
        geos_pairs.emplace_back(cand_build_indices[k], cand_probe_indices[k]);
    }

    std::sort(gpu_pairs.begin(), gpu_pairs.end());
    std::sort(geos_pairs.begin(), geos_pairs.end());

    std::cout << std::left << std::setw(11) << PredicateToString(pred)
              << " | Hits: " << std::setw(5) << gpu_pairs.size()
              << " | GPU Eval: " << std::setw(5) << gpu_eval_ms << " ms"
              << " (E2E: " << std::setw(5) << gpu_refine_ms << " ms)"
              << " | GEOS Prep(MT): " << std::setw(6) << geos_refine_ms << " ms ("
              << (geos_refine_ms / std::max(0.001, gpu_eval_ms)) << "x)\n";

    EXPECT_EQ(gpu_pairs, geos_pairs)
        << "GPU RT vs GEOS mismatch on " << PredicateToString(pred);
  }
  std::cout << "====================================================================\n";
}

TEST(JoinerTest, PolygonPolygonMultiThreadBenchmarkAndParity) {
  using namespace TestUtils;

  // Suite 1: Synthetic Polygons (10,000 x 10,000 ~7-vertex polygons)
  {
    auto poly1_path = ResolveSyntheticPolyPath("synthetic_poly/polygons1.parquet");
    auto poly2_path = ResolveSyntheticPolyPath("synthetic_poly/polygons2.parquet");

    auto poly1_arrays = ReadParquet(poly1_path, 10000);
    auto poly2_arrays = ReadParquet(poly2_path, 10000);

    std::vector<nanoarrow::UniqueArray> build_uniq_arrays, probe_uniq_arrays;
    std::vector<nanoarrow::UniqueSchema> build_uniq_schema, probe_uniq_schema;
    for (auto& arr : poly1_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, build_uniq_arrays.emplace_back().get(),
                                            build_uniq_schema.emplace_back().get()));
    }
    for (auto& arr : poly2_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, probe_uniq_arrays.emplace_back().get(),
                                            probe_uniq_schema.emplace_back().get()));
    }
    RunPolygonBenchmarkSuite("Synthetic ~7-Vertex Polygons", build_uniq_schema[0].get(),
                             build_uniq_arrays[0].get(), probe_uniq_schema[0].get(),
                             probe_uniq_arrays[0].get());
  }

  // Suite 2: Complex Polygons (600 x 600 256-vertex polygons with varying radii)
  {
    auto make_wavy_poly_wkt = [](double cx, double cy, double base_r, int n_verts,
                                 double phase) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6) << "POLYGON ((";
      double first_x = 0.0, first_y = 0.0;
      for (int k = 0; k < n_verts; ++k) {
        double angle = 2.0 * M_PI * k / n_verts;
        double r = base_r * (1.0 + 0.15 * std::sin(7.0 * angle + phase));
        double x = cx + r * std::cos(angle);
        double y = cy + r * std::sin(angle);
        if (k == 0) {
          first_x = x;
          first_y = y;
        }
        oss << x << " " << y << ", ";
      }
      oss << first_x << " " << first_y << "))";
      return oss.str();
    };

    std::vector<std::string> build_wkts, probe_wkts;
    const int grid_n = 25;  // 25 x 25 = 625 complex polygons per side
    const int n_verts = 256;
    for (int gx = 0; gx < grid_n; ++gx) {
      for (int gy = 0; gy < grid_n; ++gy) {
        double cx = gx * 10.0;
        double cy = gy * 10.0;
        build_wkts.push_back(make_wavy_poly_wkt(cx, cy, 8.0, n_verts, 0.1 * (gx + gy)));
        // Half of probes are smaller (some contained/within), half are shifted/overlapping
        if ((gx + gy) % 4 == 0) {
          probe_wkts.push_back(
              make_wavy_poly_wkt(cx, cy, 4.0, n_verts, 0.2 * (gx + gy)));
        } else {
          probe_wkts.push_back(make_wavy_poly_wkt(cx + 4.5, cy + 3.5, 7.5, n_verts,
                                                  0.3 * (gx + gy)));
        }
      }
    }

    nanoarrow::UniqueArrayStream build_stream, probe_stream;
    ArrayStreamFromWKT({build_wkts}, GEOARROW_TYPE_WKB, build_stream.get());
    ArrayStreamFromWKT({probe_wkts}, GEOARROW_TYPE_WKB, probe_stream.get());

    nanoarrow::UniqueSchema build_schema, probe_schema;
    nanoarrow::UniqueArray build_arr, probe_arr;
    ArrowError error;
    ASSERT_EQ(ArrowArrayStreamGetSchema(build_stream.get(), build_schema.get(), &error),
              NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStreamGetNext(build_stream.get(), build_arr.get(), &error),
              NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStreamGetSchema(probe_stream.get(), probe_schema.get(), &error),
              NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStreamGetNext(probe_stream.get(), probe_arr.get(), &error),
              NANOARROW_OK);

    RunPolygonBenchmarkSuite("Complex 256-Vertex Polygons", build_schema.get(),
                             build_arr.get(), probe_schema.get(), probe_arr.get());
  }
}

}  // namespace gpuspatial
