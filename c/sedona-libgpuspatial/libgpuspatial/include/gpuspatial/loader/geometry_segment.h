#ifndef GPUSPATIAL_LOADER_GEOMETRY_ARRAY_H
#define GPUSPATIAL_LOADER_GEOMETRY_ARRAY_H
#include <cassert>
#include <vector>

#include "geoarrow/geoarrow.hpp"
#include "gpuspatial/geom/box.cuh"
#include "gpuspatial/geom/geometry_collection.cuh"
#include "gpuspatial/geom/geometry_type.cuh"
#include "gpuspatial/loader/device_geometries.cuh"
#include "gpuspatial/utils/markers.hpp"
#include "gpuspatial/utils/mem_utils.hpp"
#include "gpuspatial/utils/pinned_vector.h"

#include <thrust/scan.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace gpuspatial {
namespace detail {
static uint64_t bswap_64(uint64_t x) {
  return (((x & 0xFFULL) << 56) | ((x & 0xFF00ULL) << 40) | ((x & 0xFF0000ULL) << 24) |
          ((x & 0xFF000000ULL) << 8) | ((x & 0xFF00000000ULL) >> 8) |
          ((x & 0xFF0000000000ULL) >> 24) | ((x & 0xFF000000000000ULL) >> 40) |
          ((x & 0xFF00000000000000ULL) >> 56));
}

template <typename POINT_T>
static void CheckDimension(uint8_t dimensions) {
#ifndef NDEBUG
  constexpr static int n_dim = POINT_T::n_dim;
  switch ((GeoArrowDimensions)dimensions) {
    case GEOARROW_DIMENSIONS_XY:
      assert(n_dim == 2);
      break;
    case GEOARROW_DIMENSIONS_XYZ:
      assert(n_dim == 3);
      break;
    case GEOARROW_DIMENSIONS_XYZM:
      assert(n_dim == 4);
      break;
    default:
      assert(false);
  }
#endif
}

}  // namespace detail

/**
 * An array of geometries in the same type
 */
class GeometrySegment {
 public:
  GeometrySegment() = default;
  virtual ~GeometrySegment() {}

  virtual GeometryType GetType() const = 0;

  virtual void Reserve(size_t n_features) = 0;
  // do not use virtual Add method to avoid overheads

  // return number of features
  virtual size_t size() const = 0;

  virtual void Clear() = 0;
};

template <typename POINT_T>
class PointSegment : public GeometrySegment {
 public:
  using point_t = POINT_T;
  constexpr static int n_dim = point_t::n_dim;

  void Add(const point_t* point) {
    if (point == nullptr) {
      point_t empty_point;
      empty_point.set_empty();
      points_.push_back(empty_point);
    } else {
      points_.push_back(*point);
    }
  }

  void Add(const GeoArrowGeometryView* geom) {
    if (geom == nullptr) {
      point_t empty_point;
      empty_point.set_empty();
      points_.push_back(empty_point);
    } else {
      auto* node = geom->root;
      assert(node->geometry_type == GEOARROW_GEOMETRY_TYPE_POINT);
      addPoint(node);
    }
  }

  size_t size() const override { return points_.size(); }

  void Reserve(size_t n) override { points_.reserve(n); }

  void Clear() override { points_.clear(); }

  GeometryType GetType() const override { return GeometryType::kPoint; }

  template <typename INDEX_T>
  static std::shared_ptr<DeviceGeometries<point_t, INDEX_T>> LoadOnDevice(
      const rmm::cuda_stream_view& stream,
      const std::vector<std::shared_ptr<GeometrySegment>>& segments) {
    RangeMarker marker(true, "LoadPointSegment");
    size_t n_points = 0;

    for (uint32_t segment_idx = 0; segment_idx < segments.size(); segment_idx++) {
      const auto& segment = dynamic_cast<PointSegment&>(*segments[segment_idx]);

      n_points += segment.points_.size();
    }

    size_t offset = 0;
    rmm::device_uvector<point_t> points(n_points, stream);

    for (uint32_t segment_idx = 0; segment_idx < segments.size(); segment_idx++) {
      const auto& segment = dynamic_cast<PointSegment&>(*segments[segment_idx]);

      detail::async_copy_h2d(stream, segment.points_.data(), points.data() + offset,
                             segment.points_.size());

      offset += segment.points_.size();
    }

    auto geometries = std::make_shared<DeviceGeometries<point_t, INDEX_T>>();

    geometries->type_ = GeometryType::kPoint;
    geometries->points_ = std::move(points);

    return geometries;
  }

 private:
  PinnedVector<point_t> points_;

  void addPoint(const GeoArrowGeometryNode* point_node) {
    detail::CheckDimension<point_t>(point_node->dimensions);
    assert(point_node->geometry_type == GEOARROW_GEOMETRY_TYPE_POINT);

    bool swap_endian = (point_node->flags & GEOARROW_GEOMETRY_NODE_FLAG_SWAP_ENDIAN);
    point_t new_point;

    for (int dim = 0; dim < n_dim; ++dim) {
      uint64_t coord_int;
      memcpy(&coord_int, point_node->coords[dim], sizeof(uint64_t));

      if (swap_endian) {
        coord_int = detail::bswap_64(coord_int);
      }

      double coord_double;
      memcpy(&coord_double, &coord_int, sizeof(double));

      new_point.set_coordinate(dim, coord_double);
    }
    points_.push_back(new_point);
  }
};

template <typename POINT_T, typename INDEX_T>
class MultiPointSegment : public GeometrySegment {
 public:
  using point_t = POINT_T;
  using index_t = INDEX_T;
  using box_t = Box<Point<float, POINT_T::n_dim>>;
  using dev_geometries_t = DeviceGeometries<point_t, index_t>;
  constexpr static int n_dim = point_t::n_dim;

  void Add(const GeoArrowGeometryView* geom) {
    box_t box;
    box.set_empty();
    if (geom == nullptr) {
      num_points_.push_back(0);
    } else {
      auto* node = geom->root;
      assert(node->geometry_type == GEOARROW_GEOMETRY_TYPE_POINT ||
             node->geometry_type == GEOARROW_GEOMETRY_TYPE_MULTIPOINT);

      if (node->geometry_type == GEOARROW_GEOMETRY_TYPE_POINT) {
        auto point = parsePoint(node);
        box.Expand(point);
        points_.push_back(point);
        num_points_.push_back(1);
      } else {
        parseMultiPoint(node, box);
      }
    }
    mbrs_.push_back(box);
  }

  void Reserve(size_t n) override { points_.reserve(n); }

  size_t size() const override { return num_points_.size(); }

  void Clear() override {
    num_points_.clear();
    points_.clear();
    mbrs_.clear();
  }

  GeometryType GetType() const override { return GeometryType::kMultiPoint; }

  static std::shared_ptr<dev_geometries_t> LoadOnDevice(
      const rmm::cuda_stream_view& stream,
      const std::vector<std::shared_ptr<GeometrySegment>>& segments) {
    RangeMarker marker(true, "MultiPointSegment");
    size_t size_num_points = 0;
    size_t n_points = 0;
    size_t max_seg_size = 0;

    for (size_t segment_idx = 0; segment_idx < segments.size(); segment_idx++) {
      const auto& segment = dynamic_cast<MultiPointSegment&>(*segments[segment_idx]);

      size_num_points += segment.num_points_.size();
      n_points += segment.points_.size();
      max_seg_size = std::max(max_seg_size, segment.num_points_.size());
    }

    rmm::device_uvector<point_t> points(n_points, stream);
    rmm::device_uvector<box_t> boxes(size_num_points, stream);
    index_t init_prefix_sum = 0;
    size_t write_offset_boxes = 0;
    size_t write_offset_prefix_sum = 1;
    size_t write_offset_points = 0;

    rmm::device_uvector<index_t> tmp_buf(max_seg_size, stream);
    rmm::device_uvector<INDEX_T> prefix_sum(size_num_points + 1, stream);

    prefix_sum.set_element_to_zero_async(0, stream);

    for (uint32_t segment_idx = 0; segment_idx < segments.size(); segment_idx++) {
      const auto& segment = dynamic_cast<MultiPointSegment&>(*segments[segment_idx]);

      detail::async_copy_h2d(stream, segment.points_.data(),
                             points.data() + write_offset_points, segment.points_.size());

      detail::async_copy_h2d(stream, segment.mbrs_.data(),
                             boxes.data() + write_offset_boxes, segment.mbrs_.size());

      detail::async_copy_h2d(stream, segment.num_points_.data(), tmp_buf.data(),
                             segment.num_points_.size());

      thrust::inclusive_scan(rmm::exec_policy_nosync(stream), tmp_buf.begin(),
                             tmp_buf.begin() + segment.num_points_.size(),
                             prefix_sum.begin() + write_offset_prefix_sum,
                             init_prefix_sum, thrust::plus<index_t>());

      write_offset_boxes += segment.num_points_.size();
      write_offset_prefix_sum += segment.num_points_.size();
      write_offset_points += segment.points_.size();

      if (segment_idx != segments.size() - 1) {
        init_prefix_sum = prefix_sum.element(write_offset_prefix_sum - 1, stream);
      }
    }

    auto geometries = std::make_shared<dev_geometries_t>();

    geometries->type_ = GeometryType::kMultiPoint;
    geometries->offsets_.multi_point_offsets.ps_num_points = std::move(prefix_sum);
    geometries->points_ = std::move(points);
    geometries->mbrs_ = std::move(boxes);

    return geometries;
  }

 private:
  std::vector<index_t> num_points_;
  std::vector<point_t> points_;
  std::vector<box_t> mbrs_;

  void parseMultiPoint(const GeoArrowGeometryNode* node, box_t& box) {
    for (uint32_t i = 0; i < node->size; i++) {
      auto point_node = node + i + 1;
      auto point = parsePoint(point_node);
      points_.push_back(point);
      box.Expand(point);
    }
    num_points_.push_back(node->size);
  }

  point_t parsePoint(const GeoArrowGeometryNode* point_node) const {
    detail::CheckDimension<point_t>(point_node->dimensions);
    point_t point;
    for (int dim = 0; dim < n_dim; dim++) {
      uint64_t coord_int;
      double coord_double;

      coord_int = *reinterpret_cast<const uint64_t*>(point_node->coords[dim]);
      if (point_node->flags & GEOARROW_GEOMETRY_NODE_FLAG_SWAP_ENDIAN) {
        coord_int = detail::bswap_64(coord_int);
      }
      coord_double = *reinterpret_cast<double*>(&coord_int);
      point.set_coordinate(dim, coord_double);
    }
    return point;
  }
};

template <typename POINT_T, typename INDEX_T>
class LineStringSegment : public GeometrySegment {
  using point_t = POINT_T;
  using index_t = INDEX_T;
  using box_t = Box<Point<float, POINT_T::n_dim>>;
  using dev_geometries_t = DeviceGeometries<point_t, index_t>;
  constexpr static int n_dim = point_t::n_dim;

 public:
  void Add(const GeoArrowGeometryView* geom) {
    box_t box;

    box.set_empty();
    if (geom == nullptr) {
      num_points_.push_back(0);
    } else {
      auto* node = geom->root;
      parseLineString(node, box);
    }
    mbrs_.push_back(box);
  }

  size_t size() const override { return num_points_.size(); }

  void Reserve(size_t n) override { points_.reserve(n); }

  void Clear() override {
    num_points_.clear();
    points_.clear();
    mbrs_.clear();
  }

  GeometryType GetType() const override { return GeometryType::kLineString; }

  static std::shared_ptr<dev_geometries_t> LoadOnDevice(
      const rmm::cuda_stream_view& stream,
      const std::vector<std::shared_ptr<GeometrySegment>>& segments) {
    RangeMarker marker(true, "LineStringSegment");
    size_t size_num_points = 0;
    size_t n_points = 0;
    size_t max_seg_size = 0;

    for (uint32_t segment_idx = 0; segment_idx < segments.size(); segment_idx++) {
      const auto& segment = dynamic_cast<LineStringSegment&>(*segments[segment_idx]);

      size_num_points += segment.num_points_.size();
      n_points += segment.points_.size();
      max_seg_size = std::max(max_seg_size, segment.num_points_.size());
    }

    rmm::device_uvector<point_t> points(n_points, stream);
    rmm::device_uvector<box_t> boxes(size_num_points, stream);
    index_t init_prefix_sum = 0;
    size_t write_offset_boxes = 0;
    size_t write_offset_prefix_sum = 1;
    size_t write_offset_points = 0;

    rmm::device_uvector<index_t> tmp_buf(max_seg_size, stream);
    auto prefix_sum =
        std::make_unique<rmm::device_uvector<index_t>>(size_num_points + 1, stream);

    prefix_sum->set_element_to_zero_async(0, stream);

    for (size_t segment_idx = 0; segment_idx < segments.size(); segment_idx++) {
      const auto& segment = dynamic_cast<LineStringSegment&>(*segments[segment_idx]);

      detail::async_copy_h2d(stream, segment.points_.data(),
                             points.data() + write_offset_points, segment.points_.size());

      detail::async_copy_h2d(stream, segment.mbrs_.data(),
                             boxes.data() + write_offset_boxes, segment.mbrs_.size());

      detail::async_copy_h2d(stream, segment.num_points_.data(), tmp_buf.data(),
                             segment.num_points_.size());

      thrust::inclusive_scan(rmm::exec_policy_nosync(stream), tmp_buf.begin(),
                             tmp_buf.begin() + segment.num_points_.size(),
                             prefix_sum->begin() + write_offset_prefix_sum,
                             init_prefix_sum, thrust::plus<INDEX_T>());

      write_offset_boxes += segment.num_points_.size();
      write_offset_prefix_sum += segment.num_points_.size();
      write_offset_points += segment.points_.size();

      if (segment_idx != segments.size() - 1) {
        init_prefix_sum = prefix_sum.element(write_offset_prefix_sum - 1, stream);
      }
    }

    auto geometries = std::make_shared<dev_geometries_t>();

    geometries->type_ = GeometryType::kLineString;
    geometries->offsets_.line_string_offsets.ps_num_points = std::move(prefix_sum);
    geometries->points_ = std::move(points);
    geometries->mbrs_ = std::move(boxes);

    return geometries;
  }

 private:
  std::vector<index_t> num_points_;
  std::vector<point_t> points_;
  std::vector<box_t> mbrs_;

  void parseLineString(const GeoArrowGeometryNode* node, box_t& box) {
    detail::CheckDimension<point_t>(node->dimensions);

    const uint8_t* p_coord[n_dim];
    int32_t d_coord[n_dim];

    for (int dim = 0; dim < n_dim; dim++) {
      p_coord[dim] = node->coords[dim];
      d_coord[dim] = node->coord_stride[dim];
    }

    for (uint32_t j = 0; j < node->size; j++) {
      point_t point;

      for (int dim = 0; dim < n_dim; dim++) {
        auto* coord = p_coord[dim];
        uint64_t coord_int;
        double coord_double;

        coord_int = *reinterpret_cast<const uint64_t*>(coord);
        if (node->flags & GEOARROW_GEOMETRY_NODE_FLAG_SWAP_ENDIAN) {
          coord_int = detail::bswap_64(coord_int);
        }
        coord_double = *reinterpret_cast<double*>(&coord_int);
        point.set_coordinate(dim, coord_double);
        p_coord[dim] += d_coord[dim];
      }
      points_.push_back(point);
      box.Expand(point);
    }
    num_points_.push_back(node->size);
  }
};

template <typename POINT_T, typename INDEX_T>
class MultiLineStringSegment : public GeometrySegment {
  using point_t = POINT_T;
  using index_t = INDEX_T;
  using box_t = Box<point_t>;
  using dev_geometries_t = DeviceGeometries<point_t, index_t>;
  constexpr static int n_dim = point_t::n_dim;

 public:
  void Add(const GeoArrowGeometryView* geom) {
    box_t box;

    box.set_empty();
    if (geom == nullptr) {
      num_parts_.push_back(0);
    } else {
      auto* node = geom->root;
      parseMultiLineString(node, box);
    }
    mbrs_.push_back(box);
  }

  size_t size() const override { return num_parts_.size(); }

  void Reserve(size_t n) override {
    num_parts_.reserve(n);
    mbrs_.reserve(n);
  }

  void Clear() override {
    num_parts_.clear();
    num_points_.clear();
    points_.clear();
    mbrs_.clear();
  }

  GeometryType GetType() const override { return GeometryType::kLineString; }

  static std::shared_ptr<dev_geometries_t> LoadOnDevice(
      const rmm::cuda_stream_view& stream,
      const std::vector<std::shared_ptr<GeometrySegment>>& segments) {
    RangeMarker marker(true, "LineStringSegment");
    size_t size_num_parts = 0;
    size_t size_num_points = 0;
    size_t n_points = 0;
    size_t max_seg_size = 0;

    for (uint32_t segment_idx = 0; segment_idx < segments.size(); segment_idx++) {
      const auto& segment = dynamic_cast<MultiLineStringSegment&>(*segments[segment_idx]);

      size_num_parts += segment.num_parts_.size();
      size_num_points += segment.num_points_.size();
      n_points += segment.points_.size();
      max_seg_size = std::max(
          max_seg_size, std::max(segment.num_parts_.size(), segment.num_points_.size()));
    }

    rmm::device_uvector<point_t> points(n_points, stream);
    rmm::device_uvector<box_t> boxes(size_num_parts, stream);
    index_t init_prefix_sum_geoms = 0;
    index_t init_prefix_sum_parts = 0;
    size_t write_offset_boxes = 0;
    size_t write_offset_prefix_sum_geoms = 1;
    size_t write_offset_prefix_sum_parts = 1;
    size_t write_offset_points = 0;

    rmm::device_uvector<index_t> tmp_buf(max_seg_size, stream);
    rmm::device_uvector<index_t> prefix_sum_geoms(size_num_parts + 1, stream);
    rmm::device_uvector<index_t> prefix_sum_parts(size_num_points + 1, stream);

    prefix_sum_geoms.set_element_to_zero_async(0, stream);
    prefix_sum_parts.set_element_to_zero_async(0, stream);

    for (size_t segment_idx = 0; segment_idx < segments.size(); segment_idx++) {
      const auto& segment = dynamic_cast<MultiLineStringSegment&>(*segments[segment_idx]);

      detail::async_copy_h2d(stream, segment.points_.data(),
                             points->data() + write_offset_points,
                             segment.points_.size());

      detail::async_copy_h2d(stream, segment.mbrs_.data(),
                             boxes->data() + write_offset_boxes, segment.mbrs_.size());

      detail::async_copy_h2d(stream, segment.num_parts_.data(), tmp_buf.data(),
                             segment.num_parts_.size());

      thrust::inclusive_scan(rmm::exec_policy_nosync(stream), tmp_buf.begin(),
                             tmp_buf.begin() + segment.num_parts_.size(),
                             prefix_sum_geoms->begin() + write_offset_prefix_sum_geoms,
                             init_prefix_sum_geoms, thrust::plus<index_t>());

      detail::async_copy_h2d(stream, segment.num_points_.data(), tmp_buf.data(),
                             segment.num_points_.size());

      thrust::inclusive_scan(rmm::exec_policy_nosync(stream), tmp_buf.begin(),
                             tmp_buf.begin() + segment.num_points_.size(),
                             prefix_sum_parts->begin() + write_offset_prefix_sum_parts,
                             init_prefix_sum_parts, thrust::plus<index_t>());

      write_offset_boxes += segment.num_parts_.size();
      write_offset_prefix_sum_geoms += segment.num_parts_.size();
      write_offset_prefix_sum_parts += segment.num_points_.size();
      write_offset_points += segment.points_.size();

      if (segment_idx != segments.size() - 1) {
        init_prefix_sum_geoms =
            prefix_sum_geoms->element(write_offset_prefix_sum_geoms - 1, stream);
        init_prefix_sum_parts =
            prefix_sum_parts->element(write_offset_prefix_sum_parts - 1, stream);
      }
    }

    auto geometries = std::make_shared<dev_geometries_t>();

    geometries->type_ = GeometryType::kMultiLineString;
    geometries->offsets_.multi_line_string_offsets.ps_num_parts =
        std::move(prefix_sum_geoms);
    geometries->offsets_.multi_line_string_offsets.ps_num_points =
        std::move(prefix_sum_parts);
    geometries->points_ = std::move(points);
    geometries->mbrs_ = std::move(boxes);

    return geometries;
  }

 private:
  std::vector<index_t> num_parts_;
  std::vector<index_t> num_points_;
  std::vector<point_t> points_;
  std::vector<box_t> mbrs_;  // MBR for each multi-line string

  void parseMultiLineString(const GeoArrowGeometryNode* node, box_t& box) {
    for (uint32_t j = 0; j < node->size; j++) {
      auto* part_node = node + j + 1;
      parseLineString(part_node, box);
    }
    num_parts_.push_back(node->size);
  }

  void parseLineString(const GeoArrowGeometryNode* node, box_t& box) {
    detail::CheckDimension<point_t>(node->dimensions);

    const uint8_t* p_coord[n_dim];
    int32_t d_coord[n_dim];

    for (int dim = 0; dim < n_dim; dim++) {
      p_coord[dim] = node->coords[dim];
      d_coord[dim] = node->coord_stride[dim];
    }

    for (uint32_t j = 0; j < node->size; j++) {
      point_t point;

      for (int dim = 0; dim < n_dim; dim++) {
        auto* coord = p_coord[dim];
        uint64_t coord_int;
        double coord_double;

        coord_int = *reinterpret_cast<const uint64_t*>(coord);
        if (node->flags & GEOARROW_GEOMETRY_NODE_FLAG_SWAP_ENDIAN) {
          coord_int = detail::bswap_64(coord_int);
        }
        coord_double = *reinterpret_cast<double*>(&coord_int);
        point.set_coordinate(dim, coord_double);
        p_coord[dim] += d_coord[dim];
      }
      box.Expand(point);
      points_.push_back(point);
    }
    num_points_.push_back(node->size);
  }
};

/**
 * Representing an array of polygons or multi-polygons
 * @tparam POINT_T
 */
template <typename POINT_T, typename INDEX_T>
class PolygonSegment : public GeometrySegment {
  using point_t = POINT_T;
  using index_t = INDEX_T;
  using box_t = Box<point_t>;
  using dev_geometries_t = DeviceGeometries<point_t, index_t>;
  constexpr static int n_dim = point_t::n_dim;

 public:
  void Clear() override {
    num_rings_.clear();
    num_vertices_.clear();
    mbrs_.clear();
    points_.clear();
  }

  size_t size() const override { return num_rings_.size(); }

  void ReservePoints(size_t n) { points_.reserve(n); }

  void Reserve(size_t n) override {
    num_rings_.reserve(n + 1);
    mbrs_.reserve(n);
  }

  void Add(const GeoArrowGeometryView* geom) {
    box_t box;

    if (geom == nullptr) {
      box.set_empty();
      num_rings_.push_back(0);
    } else {
      auto* node = geom->root;

      // CheckGeometryType(node, GEOARROW_GEOMETRY_TYPE_POLYGON);
      addPolygon(node, box);
    }
    mbrs_.push_back(box);
  }

  GeometryType GetType() const override { return GeometryType::kPolygon; }

  static std::shared_ptr<dev_geometries_t> LoadOnDevice(
      const rmm::cuda_stream_view& stream,
      const std::vector<std::shared_ptr<GeometrySegment>>& segments) {
    RangeMarker marker(true, "PolygonSegment");
    size_t num_points = 0;
    size_t num_boxes = 0;
    // for prefix sum
    size_t size_num_rings = 0;
    size_t size_num_vertices = 0;
    size_t max_seg_size = 0;

    for (size_t segment_idx = 0; segment_idx < segments.size(); segment_idx++) {
      const auto& segment = dynamic_cast<PolygonSegment&>(*segments[segment_idx]);
      num_points += segment.points_.size();
      num_boxes += segment.mbrs_.size();
      size_num_rings += segment.num_rings_.size();
      size_num_vertices += segment.num_vertices_.size();
      max_seg_size = std::max(max_seg_size, std::max(segment.num_vertices_.size(),
                                                     segment.num_rings_.size()));
    }

    rmm::device_uvector<index_t> tmp_buf(max_seg_size, stream);

    rmm::device_uvector<index_t> prefix_sum_polygons(size_num_rings + 1, stream);
    rmm::device_uvector<index_t> prefix_sum_rings(size_num_vertices + 1, stream);
    rmm::device_uvector<point_t> vertices(num_points, stream);
    rmm::device_uvector<box_t> boxes(num_boxes, stream);

    prefix_sum_polygons.set_element_to_zero_async(0, stream);
    prefix_sum_rings.set_element_to_zero_async(0, stream);

    size_t write_offset_points = 0;
    size_t write_offset_boxes = 0;
    size_t write_offset_parts = 1;
    size_t write_offset_rings = 1;

    index_t init_prefix_sum_parts = 0;
    index_t init_prefix_sum_rings = 0;

    for (size_t segment_idx = 0; segment_idx < segments.size(); segment_idx++) {
      const auto& segment = dynamic_cast<PolygonSegment&>(*segments[segment_idx]);

      detail::async_copy_h2d(stream, segment.points_.data(),
                             vertices.data() + write_offset_points,
                             segment.points_.size());

      detail::async_copy_h2d(stream, segment.mbrs_.data(),
                             boxes->data() + write_offset_boxes, segment.mbrs_.size());

      detail::async_copy_h2d(stream, segment.num_rings_.data(), tmp_buf.data(),
                             segment.num_rings_.size());

      thrust::inclusive_scan(rmm::exec_policy_nosync(stream), tmp_buf.begin(),
                             tmp_buf.begin() + segment.num_rings_.size(),
                             prefix_sum_polygons.begin() + write_offset_parts,
                             init_prefix_sum_parts, thrust::plus<index_t>());

      detail::async_copy_h2d(stream, segment.num_vertices_.data(), tmp_buf.data(),
                             segment.num_vertices_.size());
      thrust::inclusive_scan(rmm::exec_policy_nosync(stream), tmp_buf.begin(),
                             tmp_buf.begin() + segment.num_vertices_.size(),
                             prefix_sum_rings.begin() + write_offset_rings,
                             init_prefix_sum_rings, thrust::plus<index_t>());

      write_offset_points += segment.points_.size();
      write_offset_boxes += segment.mbrs_.size();
      write_offset_parts += segment.num_rings_.size();
      write_offset_rings += segment.num_vertices_.size();

      if (segment_idx != segments.size() - 1) {
        init_prefix_sum_parts =
            prefix_sum_polygons.element(write_offset_parts - 1, stream);
        init_prefix_sum_rings = prefix_sum_rings.element(write_offset_rings - 1, stream);
      }
    }

    auto geometries = std::make_shared<dev_geometries_t>();

    geometries->type_ = GeometryType::kPolygon;
    geometries->points_ = std::move(vertices);
    geometries->mbrs_ = std::move(boxes);
    geometries->offsets_.polygon_offsets.ps_num_rings = std::move(prefix_sum_polygons);
    geometries->offsets_.polygon_offsets.ps_num_points = std::move(prefix_sum_rings);

    return geometries;
  }

 private:
  std::vector<index_t> num_rings_;     // number of rings each polygon
  std::vector<index_t> num_vertices_;  // number of vertices each ring
  std::vector<point_t> points_;
  std::vector<box_t> mbrs_;  // MBR for each polygon/multipolygon

  template <typename FUNC_T>
  void parseRing(const GeoArrowGeometryNode* ring_node, const FUNC_T& func) {
    assert(ring_node->geometry_type == GEOARROW_GEOMETRY_TYPE_LINESTRING);
    detail::CheckDimension<point_t>(ring_node->dimensions);

    const uint8_t* p_coord[n_dim];
    int32_t d_coord[n_dim];

    for (int dim = 0; dim < n_dim; dim++) {
      p_coord[dim] = ring_node->coords[dim];
      d_coord[dim] = ring_node->coord_stride[dim];
    }

    // visit points in the ring
    assert(ring_node->size > 0);
    for (uint32_t j = 0; j < ring_node->size; j++) {
      point_t point;

      for (int dim = 0; dim < n_dim; dim++) {
        auto* coord = p_coord[dim];
        uint64_t coord_int;
        double coord_double;

        coord_int = *reinterpret_cast<const uint64_t*>(coord);
        if (ring_node->flags & GEOARROW_GEOMETRY_NODE_FLAG_SWAP_ENDIAN) {
          coord_int = detail::bswap_64(coord_int);
        }
        coord_double = *reinterpret_cast<double*>(&coord_int);
        point.set_coordinate(dim, coord_double);
        p_coord[dim] += d_coord[dim];
      }
      func(point);
    }
  }

  void addPolygon(const GeoArrowGeometryNode* polygon_node, box_t& box) {
    // visit rings
    for (uint32_t i = 0; i < polygon_node->size; i++) {
      auto ring_node = polygon_node + i + 1;

      this->parseRing(ring_node, [&](const point_t& point) {
        points_.push_back(point);
        box.Expand(point);
      });

      num_vertices_.push_back(ring_node->size);
    }
    num_rings_.push_back(polygon_node->size);
  }
};

/**
 * Representing an array of polygons or multi-polygons
 * @tparam POINT_T
 */
template <typename POINT_T, typename INDEX_T>
class MultiPolygonSegment : public GeometrySegment {
  using point_t = POINT_T;
  using index_t = INDEX_T;
  using box_t = Box<point_t>;
  using dev_geometries_t = DeviceGeometries<point_t, index_t>;
  constexpr static int n_dim = point_t::n_dim;

 public:
  void Clear() override {
    num_parts_.clear();
    num_rings_.clear();
    num_vertices_.clear();
    mbrs_.clear();
    points_.clear();
  }

  size_t size() const override { return num_parts_.size(); }

  void ReservePoints(size_t n) { points_.reserve(n); }

  void Reserve(size_t n) override {
    num_parts_.reserve(n + 1);
    mbrs_.reserve(n);
  }

  void Add(const GeoArrowGeometryView* geom) {
    box_t box;

    if (geom == nullptr) {
      num_parts_.push_back(0);
      box.set_empty();
    } else {
      auto* node = geom->root;
      auto geom_type = node->geometry_type;

      assert(geom_type == GEOARROW_GEOMETRY_TYPE_POLYGON ||
             geom_type == GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON);
      if (geom_type == GEOARROW_GEOMETRY_TYPE_POLYGON) {
        addPolygon(node, box);
        num_parts_.push_back(1);
      } else {
        auto* end = node + geom->size_nodes;
        uint32_t num_polygons = 0;

        for (; node < end; node++) {
          if (node->geometry_type == GEOARROW_GEOMETRY_TYPE_POLYGON) {
            addPolygon(node, box);
            num_polygons++;
          }
        }
        num_parts_.push_back(num_polygons);
      }
    }
    mbrs_.push_back(box);
  }

  GeometryType GetType() const override { return GeometryType::kMultiPolygon; }

  static std::shared_ptr<dev_geometries_t> LoadOnDevice(
      const rmm::cuda_stream_view& stream,
      const std::vector<std::shared_ptr<GeometrySegment>>& segments) {
    RangeMarker marker(true, "LoadMultiPolygonSegment");
    size_t num_points = 0;
    size_t num_boxes = 0;
    // for prefix sum
    size_t size_num_parts = 0;
    size_t size_num_rings = 0;
    size_t size_num_vertices = 0;
    size_t max_seg_size = 0;

    for (size_t segment_idx = 0; segment_idx < segments.size(); segment_idx++) {
      const auto& segment = dynamic_cast<MultiPolygonSegment&>(*segments[segment_idx]);
      num_points += segment.points_.size();
      num_boxes += segment.mbrs_.size();
      size_num_parts += segment.num_parts_.size();
      size_num_rings += segment.num_rings_.size();
      size_num_vertices += segment.num_vertices_.size();
      max_seg_size = std::max(
          max_seg_size,
          std::max(segment.num_vertices_.size(),
                   std::max(segment.num_rings_.size(), segment.num_parts_.size())));
    }

    rmm::device_uvector<index_t> tmp_buf(max_seg_size, stream);

    rmm::device_uvector<index_t> prefix_sum_geoms(size_num_parts + 1, stream);
    rmm::device_uvector<index_t> prefix_sum_parts(size_num_rings + 1, stream);
    rmm::device_uvector<index_t> prefix_sum_rings(size_num_vertices + 1, stream);
    rmm::device_uvector<point_t> vertices(num_points, stream);
    rmm::device_uvector<box_t> boxes(num_boxes, stream);

    prefix_sum_geoms.set_element_to_zero_async(0, stream);
    prefix_sum_parts.set_element_to_zero_async(0, stream);
    prefix_sum_rings.set_element_to_zero_async(0, stream);

    size_t write_offset_points = 0;
    size_t write_offset_boxes = 0;
    size_t write_offset_geoms = 1;
    size_t write_offset_parts = 1;
    size_t write_offset_rings = 1;

    index_t init_prefix_sum_geoms = 0;
    index_t init_prefix_sum_parts = 0;
    index_t init_prefix_sum_rings = 0;

    for (size_t segment_idx = 0; segment_idx < segments.size(); segment_idx++) {
      const auto& segment = dynamic_cast<MultiPolygonSegment&>(*segments[segment_idx]);

      detail::async_copy_h2d(stream, segment.points_.data(),
                             vertices.data() + write_offset_points,
                             segment.points_.size());
      detail::async_copy_h2d(stream, segment.mbrs_.data(),
                             boxes->data() + write_offset_boxes, segment.mbrs_.size());

      detail::async_copy_h2d(stream, segment.num_parts_.data(), tmp_buf.data(),
                             segment.num_parts_.size());
      thrust::inclusive_scan(rmm::exec_policy_nosync(stream), tmp_buf.begin(),
                             tmp_buf.begin() + segment.num_parts_.size(),
                             prefix_sum_geoms.begin() + write_offset_geoms,
                             init_prefix_sum_geoms, thrust::plus<index_t>());

      detail::async_copy_h2d(stream, segment.num_rings_.data(), tmp_buf.data(),
                             segment.num_rings_.size());
      thrust::inclusive_scan(rmm::exec_policy_nosync(stream), tmp_buf.begin(),
                             tmp_buf.begin() + segment.num_rings_.size(),
                             prefix_sum_parts.begin() + write_offset_parts,
                             init_prefix_sum_parts, thrust::plus<index_t>());

      detail::async_copy_h2d(stream, segment.num_vertices_.data(), tmp_buf.data(),
                             segment.num_vertices_.size());
      thrust::inclusive_scan(rmm::exec_policy_nosync(stream), tmp_buf.begin(),
                             tmp_buf.begin() + segment.num_vertices_.size(),
                             prefix_sum_rings.begin() + write_offset_rings,
                             init_prefix_sum_rings, thrust::plus<index_t>());

      write_offset_points += segment.points_.size();
      write_offset_boxes += segment.mbrs_.size();
      write_offset_geoms += segment.num_parts_.size();
      write_offset_parts += segment.num_rings_.size();
      write_offset_rings += segment.num_vertices_.size();

      if (segment_idx != segments.size() - 1) {
        init_prefix_sum_geoms = prefix_sum_geoms.element(write_offset_geoms - 1, stream);
        init_prefix_sum_parts = prefix_sum_parts.element(write_offset_parts - 1, stream);
        init_prefix_sum_rings = prefix_sum_rings.element(write_offset_rings - 1, stream);
      }
    }

    auto geometries = std::make_shared<dev_geometries_t>();

    geometries->type_ = GeometryType::kMultiPolygon;
    geometries->points_ = std::move(vertices);
    geometries->mbrs_ = std::move(boxes);
    geometries->offsets_.multi_polygon_offsets.ps_num_parts = std::move(prefix_sum_geoms);
    geometries->offsets_.multi_polygon_offsets.ps_num_rings = std::move(prefix_sum_parts);
    geometries->offsets_.multi_polygon_offsets.ps_num_points =
        std::move(prefix_sum_rings);

    return geometries;
  }

 private:
  std::vector<index_t> num_parts_;     // # of parts each geom, which is one for polygon
                                       // and more than one for multipolygon
  std::vector<index_t> num_rings_;     // number of rings each part
  std::vector<index_t> num_vertices_;  // number of vertices each ring
  std::vector<point_t> points_;
  std::vector<box_t> mbrs_;  // MBR for each geometry

  template <typename FUNC_T>
  void parseRing(const GeoArrowGeometryNode* ring_node, const FUNC_T& func) {
    assert(ring_node->geometry_type == GEOARROW_GEOMETRY_TYPE_LINESTRING);
    detail::CheckDimension<point_t>(ring_node->dimensions);

    const uint8_t* p_coord[n_dim];
    int32_t d_coord[n_dim];

    for (int dim = 0; dim < n_dim; dim++) {
      p_coord[dim] = ring_node->coords[dim];
      d_coord[dim] = ring_node->coord_stride[dim];
    }

    // visit points in the ring
    for (uint32_t j = 0; j < ring_node->size; j++) {
      point_t point;

      for (int dim = 0; dim < n_dim; dim++) {
        auto* coord = p_coord[dim];
        uint64_t coord_int;
        double coord_double;

        coord_int = *reinterpret_cast<const uint64_t*>(coord);
        if (ring_node->flags & GEOARROW_GEOMETRY_NODE_FLAG_SWAP_ENDIAN) {
          coord_int = detail::bswap_64(coord_int);
        }
        coord_double = *reinterpret_cast<double*>(&coord_int);
        point.set_coordinate(dim, coord_double);
        p_coord[dim] += d_coord[dim];
      }
      func(point);
    }
  }

  void addPolygon(const GeoArrowGeometryNode* polygon_node, box_t& box) {
    // visit rings
    for (uint32_t i = 0; i < polygon_node->size; i++) {
      auto ring_node = polygon_node + i + 1;

      this->parseRing(ring_node, [&](const point_t& point) {
        points_.push_back(point);
        box.Expand(point);
      });

      num_vertices_.push_back(ring_node->size);
    }
    num_rings_.push_back(polygon_node->size);
  }
};

template <typename POINT_T, typename INDEX_T>
class GeometryCollectionSegment : public GeometrySegment {
  using point_t = POINT_T;
  using index_t = INDEX_T;
  using box_t = Box<point_t>;
  using dev_geometries_t = DeviceGeometries<point_t, index_t>;
  constexpr static int n_dim = point_t::n_dim;
  using feature_t = detail::Feature<index_t>;
  using part_t = detail::GeometryPart<index_t>;

 public:
  GeometryCollectionSegment() : type_(GeometryType::kNumGeometryTypes) {}

  size_t size() const override { return features_.size(); }

  void Reserve(size_t n) override { features_.reserve(n); }

  void Clear() {
    features_.clear();
    parts_.clear();
    ring_starts_.clear();
    vertices_.clear();
    mbrs_.clear();
    type_ = GeometryType::kNumGeometryTypes;
  }

  GeometryType GetType() const override { return type_; }

  void Add(const GeoArrowGeometryView* geom) {
    if (geom == nullptr) {
      features_.emplace_back(GeometryType::kNumGeometryTypes, parts_.size(), 0);
      mbrs_.emplace_back();
      return;
    }
    auto* node = geom->root;
    auto geom_type = (GeoArrowGeometryType)node->geometry_type;
    auto n_parts = node->size;
    box_t mbr;

    if (type_ == GeometryType::kNumGeometryTypes) {
      type_ = FromGeoArrowGeometryType(geom_type);
    } else {
      type_ = GetCompatibleGeometryType(type_, FromGeoArrowGeometryType(geom_type));
    }

    switch (geom_type) {
      case GEOARROW_GEOMETRY_TYPE_POINT: {
        features_.emplace_back(GeometryType::kPoint, parts_.size(), 1);
        addPointPart(node, mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_MULTIPOINT: {
        features_.emplace_back(GeometryType::kMultiPoint, parts_.size(), n_parts);
        for (uint32_t i = 0; i < n_parts; i++) {
          auto point_node = node + i + 1;
          addPointPart(point_node, mbr);
        }
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_LINESTRING: {
        features_.emplace_back(GeometryType::kLineString, parts_.size(), 1);
        addLineStringPart(node, mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING: {
        features_.emplace_back(GeometryType::kMultiLineString, parts_.size(), n_parts);
        for (uint32_t j = 0; j < n_parts; j++) {
          auto* part_node = node + j + 1;
          addLineStringPart(part_node, mbr);
        }
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_POLYGON: {
        features_.emplace_back(GeometryType::kPolygon, parts_.size(), 1);
        addPolygonPart(node, mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON: {
        features_.emplace_back(GeometryType::kPolygon, parts_.size(), 1);
        for (uint32_t j = 0; j < n_parts; j++) {
          auto* part_node = node + j + 1;
          if (part_node->geometry_type == GEOARROW_GEOMETRY_TYPE_POLYGON) {
            addPolygonPart(part_node, mbr);
          }
        }
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION: {
        features_.emplace_back(GeometryType::kGeometryCollection, parts_.size(), n_parts);
        for (uint32_t j = 0; j < n_parts; j++) {
          auto* part_node = node + j + 1;

          addPartRecursively(part_node, mbr);
        }
      }
    }
    mbrs_.push_back(mbr);
  }

 private:
  GeometryType type_;
  std::vector<feature_t> features_;
  std::vector<part_t> parts_;
  std::vector<index_t> ring_starts_;  // start index of each ring in the geometry
  std::vector<point_t> vertices_;
  std::vector<box_t> mbrs_;  // MBR for each geometry

  point_t parsePoint(const GeoArrowGeometryNode* point_node) const {
    detail::CheckDimension<point_t>(point_node->dimensions);
    assert(point_node->geometry_type == GEOARROW_GEOMETRY_TYPE_POINT);
    point_t point;
    for (int dim = 0; dim < n_dim; dim++) {
      uint64_t coord_int;
      double coord_double;

      coord_int = *reinterpret_cast<const uint64_t*>(point_node->coords[dim]);
      if (point_node->flags & GEOARROW_GEOMETRY_NODE_FLAG_SWAP_ENDIAN) {
        coord_int = detail::bswap_64(coord_int);
      }
      coord_double = *reinterpret_cast<double*>(&coord_int);
      point.set_coordinate(dim, coord_double);
    }
    return point;
  }

  void addPointPart(const GeoArrowGeometryNode* node, box_t& mbr) {
    parts_.emplace_back(GeometryType::kPoint, vertices_.size(), 1);
    auto point = parsePoint(node);
    vertices_.push_back(point);
    mbr = box_t{point, point};
  }

  void addLineStringPart(const GeoArrowGeometryNode* node, box_t& mbr) {
    detail::CheckDimension<point_t>(node->dimensions);

    const uint8_t* p_coord[n_dim];
    int32_t d_coord[n_dim];

    for (int dim = 0; dim < n_dim; dim++) {
      p_coord[dim] = node->coords[dim];
      d_coord[dim] = node->coord_stride[dim];
    }

    parts_.emplace_back(GeometryType::kLineString, vertices_.size(), node->size);

    for (uint32_t j = 0; j < node->size; j++) {
      point_t point;

      for (int dim = 0; dim < n_dim; dim++) {
        auto* coord = p_coord[dim];
        uint64_t coord_int;
        double coord_double;

        coord_int = *reinterpret_cast<const uint64_t*>(coord);
        if (node->flags & GEOARROW_GEOMETRY_NODE_FLAG_SWAP_ENDIAN) {
          coord_int = detail::bswap_64(coord_int);
        }
        coord_double = *reinterpret_cast<double*>(&coord_int);
        point.set_coordinate(dim, coord_double);
        p_coord[dim] += d_coord[dim];
      }
      vertices_.emplace_back(point);
      mbr.Expand(point);
    }
  }

  void addRing(const GeoArrowGeometryNode* ring_node, box_t& mbr) {
    assert(ring_node->geometry_type == GEOARROW_GEOMETRY_TYPE_LINESTRING);
    detail::CheckDimension<point_t>(ring_node->dimensions);

    const uint8_t* p_coord[n_dim];
    int32_t d_coord[n_dim];

    for (int dim = 0; dim < n_dim; dim++) {
      p_coord[dim] = ring_node->coords[dim];
      d_coord[dim] = ring_node->coord_stride[dim];
    }

    // visit points in the ring
    assert(ring_node->size > 0);
    for (uint32_t j = 0; j < ring_node->size; j++) {
      point_t point;

      for (int dim = 0; dim < n_dim; dim++) {
        auto* coord = p_coord[dim];
        uint64_t coord_int;
        double coord_double;

        coord_int = *reinterpret_cast<const uint64_t*>(coord);
        if (ring_node->flags & GEOARROW_GEOMETRY_NODE_FLAG_SWAP_ENDIAN) {
          coord_int = detail::bswap_64(coord_int);
        }
        coord_double = *reinterpret_cast<double*>(&coord_int);
        point.set_coordinate(dim, coord_double);
        p_coord[dim] += d_coord[dim];
      }
      vertices_.push_back(point);
      mbr.Expand(point);
    }
  }

  void addPolygonPart(const GeoArrowGeometryNode* node, box_t& mbr) {
    INDEX_T n_vertices = 0;

    parts_.emplace_back(GeometryType::kPolygon, vertices_.size(), 0, ring_starts_.size(),
                        node->size);

    for (uint32_t i = 0; i < node->size; i++) {
      auto ring_node = node + i + 1;

      addRing(ring_node, mbr);
      ring_starts_.push_back(n_vertices);
      n_vertices += ring_node->size;
    }
    ring_starts_.push_back(n_vertices);  // end of the last ring
    parts_.back().vertex_count = n_vertices;
  }

  void addPartRecursively(const GeoArrowGeometryNode* node, box_t& mbr) {
    auto geom_type = (GeoArrowGeometryType)node->geometry_type;
    auto n_parts = node->size;

    switch (geom_type) {
      case GEOARROW_GEOMETRY_TYPE_POINT: {
        addPointPart(node, mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_MULTIPOINT: {
        for (uint32_t i = 0; i < n_parts; i++) {
          auto point_node = node + i + 1;
          addPointPart(point_node, mbr);
        }
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_LINESTRING: {
        addLineStringPart(node, mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING: {
        for (uint32_t j = 0; j < n_parts; j++) {
          auto* part_node = node + j + 1;
          addLineStringPart(part_node, mbr);
        }
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_POLYGON: {
        addPolygonPart(node, mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON: {
        for (uint32_t j = 0; j < n_parts; j++) {
          auto* part_node = node + j + 1;
          addPolygonPart(part_node, mbr);
        }
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION: {
        for (uint32_t j = 0; j < n_parts; j++) {
          auto* part_node = node + j + 1;

          addPartRecursively(part_node, mbr);
        }
        break;
      }
    }
  }
};

template <typename POINT_T, typename INDEX_T>
class BoxSegment : public GeometrySegment {
  using point_t = POINT_T;
  using index_t = INDEX_T;
  using box_t = Box<Point<float, point_t::n_dim>>;
  using dev_geometries_t = DeviceGeometries<point_t, index_t>;
  constexpr static int n_dim = point_t::n_dim;

 public:
  void Add(const box_t* box) {
    if (box == nullptr) {
      box_t empty_box;
      empty_box.set_empty();
    } else {
      mbrs_.push_back(*box);
    }
  }

  size_t size() const override { return mbrs_.size(); }

  void Reserve(size_t n) override { mbrs_.reserve(n); }

  void Clear() override { mbrs_.clear(); }

  GeometryType GetType() const override { return GeometryType::kBox; }

  static std::shared_ptr<dev_geometries_t> LoadOnDevice(
      const rmm::cuda_stream_view& stream,
      const std::vector<std::shared_ptr<GeometrySegment>>& segments) {
    RangeMarker marker(true, "BoxSegment");
    size_t n_boxes = 0;

    for (uint32_t segment_idx = 0; segment_idx < segments.size(); segment_idx++) {
      const auto& segment = dynamic_cast<BoxSegment&>(*segments[segment_idx]);

      n_boxes += segment.mbrs_.size();
    }

    auto boxes = std::make_unique<rmm::device_uvector<box_t>>(n_boxes, stream);
    size_t write_offset = 0;

    for (size_t segment_idx = 0; segment_idx < segments.size(); segment_idx++) {
      const auto& segment = dynamic_cast<BoxSegment&>(*segments[segment_idx]);

      detail::async_copy_h2d(stream, segment.mbrs_.data(), boxes->data() + write_offset,
                             segment.mbrs_.size());

      write_offset += segment.mbrs_.size();
    }

    auto geometries = std::make_shared<dev_geometries_t>();

    geometries->type_ = GeometryType::kBox;
    geometries->mbrs_ = std::move(boxes);

    return geometries;
  }

 private:
  PinnedVector<box_t> mbrs_;
};

template <typename POINT_T, typename INDEX_T>
inline std::shared_ptr<GeometrySegment> CreateGeometrySegment(GeometryType type) {
  std::shared_ptr<GeometrySegment> seg;
  switch (type) {
    case GeometryType::kPoint: {
      seg = std::make_shared<PointSegment<POINT_T>>();
      break;
    }
    case GeometryType::kMultiPoint: {
      seg = std::make_shared<MultiPointSegment<POINT_T, INDEX_T>>();
      break;
    }
    case GeometryType::kLineString: {
      seg = std::make_shared<LineStringSegment<POINT_T, INDEX_T>>();
      break;
    }
    case GeometryType::kMultiLineString: {
      seg = std::make_shared<MultiLineStringSegment<POINT_T, INDEX_T>>();
      break;
    }
    case GeometryType::kPolygon: {
      seg = std::make_shared<PolygonSegment<POINT_T, INDEX_T>>();
      break;
    }
    case GeometryType::kMultiPolygon: {
      seg = std::make_shared<MultiPolygonSegment<POINT_T, INDEX_T>>();
      break;
    }
    case GeometryType::kGeometryCollection: {
      seg = std::make_shared<GeometryCollectionSegment<POINT_T, INDEX_T>>();
      break;
    }
    case GeometryType::kBox: {
      seg = std::make_shared<BoxSegment<POINT_T, INDEX_T>>();
      break;
    }
    default:
      assert(false);
  }
  return seg;
}

}  // namespace gpuspatial

#endif  //  GPUSPATIAL_LOADER_GEOMETRY_ARRAY_H
