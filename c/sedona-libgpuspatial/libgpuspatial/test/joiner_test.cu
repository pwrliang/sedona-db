#include "array_stream.hpp"
#include "gpuspatial/index/spatial_joiner.cuh"
#include "gpuspatial/loader/device_geometries.cuh"
#include "gpuspatial/loader/wkb_loader.h"
#include "nanoarrow/nanoarrow.hpp"

#include "geoarrow_geos/geoarrow_geos.hpp"

#include <geoarrow/geoarrow.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <numeric>  // For std::iota
namespace TestUtils {
std::string GetTestDataPath(const std::string& relative_path_to_file);
}

namespace gpuspatial {
template <typename KeyType, typename ValueType>
void sort_vectors_by_index(std::vector<KeyType>& keys, std::vector<ValueType>& values) {
  // 1. Create an index vector {0, 1, 2, ...}
  std::vector<size_t> indices(keys.size());
  // Fills 'indices' with 0, 1, 2, ..., N-1
  std::iota(indices.begin(), indices.end(), 0);

  // 2. Sort the indices based on the values in the 'keys' vector
  // The lambda compares the key elements at two different indices
  std::sort(indices.begin(), indices.end(), [&keys, &values](size_t i, size_t j) {
    return keys[i] < keys[j] || keys[i] == keys[j] && values[i] < values[j];
  });

  // 3. Create new, sorted vectors
  std::vector<KeyType> sorted_keys;
  std::vector<ValueType> sorted_values;

  for (size_t i : indices) {
    sorted_keys.push_back(keys[i]);
    sorted_values.push_back(values[i]);
  }

  // Replace the original vectors with the sorted ones
  keys = std::move(sorted_keys);
  values = std::move(sorted_values);
}

TEST(JoinerTest, PIP) {
  SpatialJoiner::SpatialJoinerConfig config;
  std::string ptx_root = TestUtils::GetTestDataPath("shaders_ptx");

  config.ptx_root = ptx_root.c_str();
  SpatialJoiner spatial_joiner;

  nanoarrow::UniqueArrayStream poly_stream, point_stream;

  auto poly_path = TestUtils::GetTestDataPath("../test_data/test_polygons.arrows");
  auto point_path = TestUtils::GetTestDataPath("../test_data/test_points.arrows");

  ArrayStreamFromIpc(poly_path, "geometry", poly_stream.get());
  ArrayStreamFromIpc(point_path, "geometry", point_stream.get());

  nanoarrow::UniqueSchema build_schema, stream_schema;
  nanoarrow::UniqueArray build_array, stream_array;
  ArrowError error;
  ArrowErrorSet(&error, "");
  int n_row_groups = 100;
  int array_index_offset = 0;
  std::vector<uint32_t> build_indices, stream_indices;
  geoarrow::geos::ArrayReader reader;

  class GEOSCppHandle {
   public:
    GEOSContextHandle_t handle;

    GEOSCppHandle() { handle = GEOS_init_r(); }

    ~GEOSCppHandle() { GEOS_finish_r(handle); }
  };
  GEOSCppHandle handle;

  reader.InitFromEncoding(handle.handle, GEOARROW_GEOS_ENCODING_WKB);

  geoarrow::geos::GeometryVector geom_polygons(handle.handle);
  geoarrow::geos::GeometryVector geom_points(handle.handle);
  struct Payload {
    GEOSContextHandle_t handle;
    const GEOSGeometry* geom;
    int64_t stream_index_offset;
    std::vector<uint32_t> build_indices;
    std::vector<uint32_t> stream_indices;
  };

  spatial_joiner.Init(&config);
  for (int i = 0; i < n_row_groups; i++) {
    ASSERT_EQ(ArrowArrayStreamGetNext(poly_stream.get(), build_array.get(), &error),
              NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStreamGetSchema(poly_stream.get(), build_schema.get(), &error),
              NANOARROW_OK);

    ASSERT_EQ(ArrowArrayStreamGetNext(point_stream.get(), stream_array.get(), &error),
              NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStreamGetSchema(point_stream.get(), stream_schema.get(), &error),
              NANOARROW_OK);

    spatial_joiner.Clear();
    spatial_joiner.PushBuild(nullptr, build_array.get(), 0, build_array->length);
    auto context = spatial_joiner.CreateContext();

    build_indices.clear();
    stream_indices.clear();
    spatial_joiner.FinishBuilding();
    spatial_joiner.PushStream(context.get(), nullptr, stream_array.get(), 0,
                              stream_array->length, Predicate::kContains, &build_indices,
                              &stream_indices, array_index_offset);

    geom_polygons.resize(build_array->length);
    geom_points.resize(stream_array->length);

    size_t n_polygons = 0, n_points = 0;
    ASSERT_EQ(reader.Read(build_array.get(), 0, build_array->length,
                          geom_polygons.mutable_data(), &n_polygons),
              GEOARROW_GEOS_OK);
    ASSERT_EQ(reader.Read(stream_array.get(), 0, stream_array->length,
                          geom_points.mutable_data(), &n_points),
              GEOARROW_GEOS_OK);

    auto* tree = GEOSSTRtree_create_r(handle.handle, 10);

    for (size_t j = 0; j < n_polygons; j++) {
      auto* geom_polygon = geom_polygons.borrow(j);
      auto* box = GEOSEnvelope_r(handle.handle, geom_polygon);
      GEOSGeom_setUserData_r(handle.handle, (GEOSGeometry*)geom_polygon, (void*)j);
      GEOSSTRtree_insert_r(handle.handle, tree, box, (void*)geom_polygon);
      GEOSGeom_destroy_r(handle.handle, box);
    }
    ASSERT_EQ(GEOSSTRtree_build_r(handle.handle, tree), 1);

    Payload payload;
    payload.handle = handle.handle;

    payload.stream_index_offset = array_index_offset;

    for (size_t j = 0; j < n_points; j++) {
      auto* geom_point = geom_points.borrow(j);
      GEOSGeom_setUserData_r(handle.handle, (GEOSGeometry*)geom_point, (void*)j);
      double x, y;

      GEOSGeomGetX_r(handle.handle, geom_point, &x);
      GEOSGeomGetY_r(handle.handle, geom_point, &y);

      payload.geom = geom_point;

      GEOSSTRtree_query_r(
          handle.handle, tree, geom_point,
          [](void* item, void* data) {
            auto* polygon = (GEOSGeometry*)item;
            auto* payload = (Payload*)data;
            auto* point = payload->geom;

            if (GEOSContains_r(payload->handle, polygon, point) == 1) {
              auto polygon_id = (size_t)GEOSGeom_getUserData_r(payload->handle, polygon);
              auto point_id = (size_t)GEOSGeom_getUserData_r(payload->handle, point);
              payload->build_indices.push_back(polygon_id);
              payload->stream_indices.push_back(payload->stream_index_offset + point_id);
            }
          },
          (void*)&payload);
    }

    GEOSSTRtree_destroy_r(handle.handle, tree);

    ASSERT_EQ(payload.build_indices.size(), build_indices.size());
    ASSERT_EQ(payload.stream_indices.size(), stream_indices.size());
    sort_vectors_by_index(payload.build_indices, payload.stream_indices);
    sort_vectors_by_index(build_indices, stream_indices);
    for (size_t j = 0; j < build_indices.size(); j++) {
      ASSERT_EQ(payload.build_indices[j], build_indices[j]);
      ASSERT_EQ(payload.stream_indices[j], stream_indices[j]);
    }

    array_index_offset += stream_array->length;
  }
}

TEST(JoinerTest, PIPInverse) {
  SpatialJoiner::SpatialJoinerConfig config;
  std::string ptx_root = TestUtils::GetTestDataPath("shaders_ptx");

  config.ptx_root = ptx_root.c_str();
  SpatialJoiner spatial_joiner;

  nanoarrow::UniqueArrayStream poly_stream, point_stream;

  auto poly_path = TestUtils::GetTestDataPath("../test_data/test_polygons.arrows");
  auto point_path = TestUtils::GetTestDataPath("../test_data/test_points.arrows");

  ArrayStreamFromIpc(poly_path, "geometry", poly_stream.get());
  ArrayStreamFromIpc(point_path, "geometry", point_stream.get());

  nanoarrow::UniqueSchema build_schema, stream_schema;
  nanoarrow::UniqueArray build_array, stream_array;
  ArrowError error;
  ArrowErrorSet(&error, "");
  int n_row_groups = 100;
  int array_index_offset = 0;
  std::vector<uint32_t> build_indices, stream_indices;
  geoarrow::geos::ArrayReader reader;

  class GEOSCppHandle {
   public:
    GEOSContextHandle_t handle;

    GEOSCppHandle() { handle = GEOS_init_r(); }

    ~GEOSCppHandle() { GEOS_finish_r(handle); }
  };
  GEOSCppHandle handle;

  reader.InitFromEncoding(handle.handle, GEOARROW_GEOS_ENCODING_WKB);

  geoarrow::geos::GeometryVector geom_polygons(handle.handle);
  geoarrow::geos::GeometryVector geom_points(handle.handle);
  struct Payload {
    GEOSContextHandle_t handle;
    const GEOSGeometry* geom;
    int64_t stream_index_offset;
    std::vector<uint32_t> build_indices;
    std::vector<uint32_t> stream_indices;
  };

  spatial_joiner.Init(&config);
  for (int i = 0; i < n_row_groups; i++) {
    ASSERT_EQ(ArrowArrayStreamGetNext(poly_stream.get(), stream_array.get(), &error),
              NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStreamGetSchema(poly_stream.get(), stream_schema.get(), &error),
              NANOARROW_OK);

    ASSERT_EQ(ArrowArrayStreamGetNext(point_stream.get(), build_array.get(), &error),
              NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStreamGetSchema(point_stream.get(), build_schema.get(), &error),
              NANOARROW_OK);

    spatial_joiner.Clear();
    // build with points, stream with polygons
    spatial_joiner.PushBuild(nullptr, build_array.get(), 0, build_array->length);
    auto context = spatial_joiner.CreateContext();

    build_indices.clear();
    stream_indices.clear();
    spatial_joiner.FinishBuilding();
    spatial_joiner.PushStream(context.get(), nullptr, stream_array.get(), 0,
                              stream_array->length, Predicate::kWithin, &build_indices,
                              &stream_indices, array_index_offset);

    geom_polygons.resize(stream_array->length);
    geom_points.resize(build_array->length);

    size_t n_polygons = 0, n_points = 0;
    ASSERT_EQ(reader.Read(build_array.get(), 0, build_array->length,
                          geom_points.mutable_data(), &n_points),
              GEOARROW_GEOS_OK);
    ASSERT_EQ(reader.Read(stream_array.get(), 0, stream_array->length,
                          geom_polygons.mutable_data(), &n_polygons),
              GEOARROW_GEOS_OK);

    auto* tree = GEOSSTRtree_create_r(handle.handle, 10);

    for (size_t j = 0; j < n_polygons; j++) {
      auto* geom_polygon = geom_polygons.borrow(j);
      auto* box = GEOSEnvelope_r(handle.handle, geom_polygon);
      GEOSGeom_setUserData_r(handle.handle, (GEOSGeometry*)geom_polygon, (void*)j);
      GEOSSTRtree_insert_r(handle.handle, tree, box, (void*)geom_polygon);
      GEOSGeom_destroy_r(handle.handle, box);
    }
    ASSERT_EQ(GEOSSTRtree_build_r(handle.handle, tree), 1);

    Payload payload;
    payload.handle = handle.handle;

    payload.stream_index_offset = array_index_offset;

    for (size_t j = 0; j < n_points; j++) {
      auto* geom_point = geom_points.borrow(j);
      GEOSGeom_setUserData_r(handle.handle, (GEOSGeometry*)geom_point, (void*)j);
      double x, y;

      GEOSGeomGetX_r(handle.handle, geom_point, &x);
      GEOSGeomGetY_r(handle.handle, geom_point, &y);

      payload.geom = geom_point;

      GEOSSTRtree_query_r(
          handle.handle, tree, geom_point,
          [](void* item, void* data) {
            auto* polygon = (GEOSGeometry*)item;
            auto* payload = (Payload*)data;
            auto* point = payload->geom;

            if (GEOSContains_r(payload->handle, polygon, point) == 1) {
              auto polygon_id = (size_t)GEOSGeom_getUserData_r(payload->handle, polygon);
              auto point_id = (size_t)GEOSGeom_getUserData_r(payload->handle, point);
              payload->build_indices.push_back(point_id);
              payload->stream_indices.push_back(payload->stream_index_offset +
                                                polygon_id);
            }
          },
          (void*)&payload);
    }

    GEOSSTRtree_destroy_r(handle.handle, tree);

    ASSERT_EQ(payload.build_indices.size(), build_indices.size());
    ASSERT_EQ(payload.stream_indices.size(), stream_indices.size());
    sort_vectors_by_index(payload.build_indices, payload.stream_indices);
    sort_vectors_by_index(build_indices, stream_indices);
    for (size_t j = 0; j < build_indices.size(); j++) {
      ASSERT_EQ(payload.build_indices[j], build_indices[j]);
      ASSERT_EQ(payload.stream_indices[j], stream_indices[j]);
    }

    array_index_offset += stream_array->length;
  }
}
#if 0
TEST(JoinerTest, PolygonPolygonContains) {
  SpatialJoiner::SpatialJoinerConfig config;
  std::string ptx_root = TestUtils::GetTestDataPath("shaders_ptx");

  config.ptx_root = ptx_root.c_str();
  SpatialJoiner spatial_joiner;

  nanoarrow::UniqueArrayStream poly1_stream, poly2_stream;

  auto poly1_path = TestUtils::GetTestDataPath("../test_data/test_polygons1.arrows");
  auto poly2_path = TestUtils::GetTestDataPath("../test_data/test_polygons2.arrows");

  ArrayStreamFromIpc(poly1_path, "geometry", poly1_stream.get());
  ArrayStreamFromIpc(poly2_path, "geometry", poly2_stream.get());

  nanoarrow::UniqueSchema build_schema, stream_schema;
  nanoarrow::UniqueArray build_array, stream_array;
  ArrowError error;
  ArrowErrorSet(&error, "");
  int n_row_groups = 100;
  int array_index_offset = 0;
  std::vector<uint32_t> build_indices, stream_indices;
  geoarrow::geos::ArrayReader reader;

  class GEOSCppHandle {
   public:
    GEOSContextHandle_t handle;

    GEOSCppHandle() { handle = GEOS_init_r(); }

    ~GEOSCppHandle() { GEOS_finish_r(handle); }
  };
  GEOSCppHandle handle;

  reader.InitFromEncoding(handle.handle, GEOARROW_GEOS_ENCODING_WKB);

  geoarrow::geos::GeometryVector geom_polygons1(handle.handle);
  geoarrow::geos::GeometryVector geom_polygons2(handle.handle);
  struct Payload {
    GEOSContextHandle_t handle;
    const GEOSGeometry* geom;
    int64_t build_index_offset;
    int64_t stream_index_offset;
    std::vector<int64_t> build_indices;
    std::vector<int64_t> stream_indices;
  };

  int64_t build_count = 0;
  spatial_joiner.Init(&config);
  for (int i = 0; i < n_row_groups; i++) {
    ASSERT_EQ(ArrowArrayStreamGetNext(poly1_stream.get(), build_array.get(), &error),
              NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStreamGetSchema(poly1_stream.get(), build_schema.get(), &error),
              NANOARROW_OK);

    ASSERT_EQ(ArrowArrayStreamGetNext(poly2_stream.get(), stream_array.get(), &error),
              NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStreamGetSchema(poly2_stream.get(), stream_schema.get(), &error),
              NANOARROW_OK);

    spatial_joiner.Clear();
    spatial_joiner.PushBuild(nullptr, build_array.get(), 0, build_array->length);
    auto context = spatial_joiner.CreateContext();

    build_indices.clear();
    stream_indices.clear();
    spatial_joiner.FinishBuilding();
    spatial_joiner.PushStream(context.get(), nullptr, stream_array.get(), 0,
                              stream_array->length, Predicate::kContains, &build_indices,
                              &stream_indices, array_index_offset);
    geom_polygons1.resize(build_array->length);
    geom_polygons2.resize(stream_array->length);

    size_t n_polygons1 = 0, n_polygons2 = 0;
    ASSERT_EQ(reader.Read(build_array.get(), 0, build_array->length,
                          geom_polygons1.mutable_data(), &n_polygons1),
              GEOARROW_GEOS_OK);
    ASSERT_EQ(reader.Read(stream_array.get(), 0, stream_array->length,
                          geom_polygons2.mutable_data(), &n_polygons2),
              GEOARROW_GEOS_OK);

    auto* tree = GEOSSTRtree_create_r(handle.handle, 10);

    for (size_t j = 0; j < n_polygons1; j++) {
      auto* geom_polygon = geom_polygons1.borrow(j);
      auto* box = GEOSEnvelope_r(handle.handle, geom_polygon);
      GEOSGeom_setUserData_r(handle.handle, (GEOSGeometry*)geom_polygon, (void*)j);
      GEOSSTRtree_insert_r(handle.handle, tree, box, (void*)geom_polygon);
      GEOSGeom_destroy_r(handle.handle, box);
    }
    ASSERT_EQ(GEOSSTRtree_build_r(handle.handle, tree), 1);

    Payload payload;
    payload.handle = handle.handle;

    payload.build_index_offset = build_count;
    payload.stream_index_offset = array_index_offset;

    for (size_t j = 0; j < n_polygons2; j++) {
      auto* geom_poly2 = geom_polygons2.borrow(j);
      GEOSGeom_setUserData_r(handle.handle, (GEOSGeometry*)geom_poly2, (void*)j);

      payload.geom = geom_poly2;

      GEOSSTRtree_query_r(
          handle.handle, tree, geom_poly2,
          [](void* item, void* data) {
            auto* polygon1 = (GEOSGeometry*)item;
            auto* payload = (Payload*)data;
            auto* polygon2 = payload->geom;

            if (GEOSContains_r(payload->handle, polygon1, polygon2) == 1) {
              auto polygon1_id =
                  (size_t)GEOSGeom_getUserData_r(payload->handle, polygon1);
              auto polygon2_id =
                  (size_t)GEOSGeom_getUserData_r(payload->handle, polygon2);
              payload->build_indices.push_back(payload->build_index_offset + polygon1_id);
              payload->stream_indices.push_back(payload->stream_index_offset +
                                                polygon2_id);
            }
          },
          (void*)&payload);
    }

    GEOSSTRtree_destroy_r(handle.handle, tree);

    ASSERT_EQ(payload.build_indices.size(), build_indices.size());

    build_count += build_array->length;
    array_index_offset += stream_array->length;
  }
}
#endif

}  // namespace gpuspatial
