#pragma once
#include <thrust/scan.h>
#include <thread>
#include <unordered_set>

#include "gpuspatial/geom/geometry_type.cuh"
#include "gpuspatial/loader/device_geometries.cuh"
#include "gpuspatial/utils/mem_utils.hpp"
#include "nanoarrow/nanoarrow.h"
#include "rmm/cuda_stream_view.hpp"
#include "rmm/device_uvector.hpp"
#include "rmm/exec_policy.hpp"
#include "rmm/mr/device/managed_memory_resource.hpp"

#include <condition_variable>
#include <mutex>
#include <thread>

namespace gpuspatial {
namespace detail {
template <typename POINT_T, typename INDEX_T>
struct HostParsedGeometries {
  constexpr static int n_dim = POINT_T::n_dim;
  using mbr_t = Box<Point<float, n_dim> >;
  INDEX_T num_features;  // num features including nulls in Arrow table
  // each feature should have only one type except GeometryCollection
  std::vector<GeometryType> feature_types;
  // Should be size of num_features
  // This number should be one except GeometryCollection, which should be unnested # of
  // geometries
  std::vector<INDEX_T> num_geoms;
  std::vector<INDEX_T> num_parts;
  std::vector<INDEX_T> num_rings;
  std::vector<INDEX_T> num_points;
  std::vector<POINT_T> vertices;
  std::vector<mbr_t> mbrs;
  bool has_mixed_types = false;
  bool has_geometry_collection = false;
  bool create_mbr = false;

  HostParsedGeometries(bool has_mixed_types_, bool has_geometry_collection_,
                       bool create_mbr_) {
    has_mixed_types = has_mixed_types_;
    has_geometry_collection = has_geometry_collection_;
    if (has_geometry_collection) {
      has_mixed_types = true;
    }
    create_mbr = create_mbr_;
  }

  void AddGeometry(const GeoArrowGeometryView* geom) {
    if (geom == nullptr) {
      // TODO
      return;
    }

    auto root = geom->root;
    // All should be one except for GeometryCollection
    uint32_t ngeoms =
        root->geometry_type == GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION ? 0 : 1;
    mbr_t mbr;
    mbr.set_empty();
    mbr_t* p_mbr = create_mbr ? &mbr : nullptr;

    switch (root->geometry_type) {
      case GEOARROW_GEOMETRY_TYPE_POINT: {
        addPoint(root, p_mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_LINESTRING: {
        addLineString(root, p_mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_POLYGON: {
        addPolygon(root, p_mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_MULTIPOINT: {
        addMultiPoint(root, p_mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING: {
        addMultiLineString(root, p_mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON: {
        addMultiPolygon(root, p_mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION: {
        // Complexity: O(size_nodes * max_depth)
        addGeometryCollection(root, geom->size_nodes, 0, p_mbr, ngeoms);
        break;
      }
    }
    if (has_geometry_collection) {
      num_geoms.push_back(ngeoms);
    }
    if (create_mbr) {
      mbrs.push_back(mbr);
    }
  }

 private:
  void addPoint(const GeoArrowGeometryNode* node, mbr_t* mbr) {
    auto point = readPoint(node);
    if (has_mixed_types) {
      feature_types.push_back(GeometryType::kPoint);
      num_parts.push_back(1);
      num_rings.push_back(1);
      num_points.push_back(1);
    }
    vertices.push_back(point);
    if (mbr != nullptr) {
      mbr->Expand(point.as_float());
    }
  }

  void addMultiPoint(const GeoArrowGeometryNode* node, mbr_t* mbr) {
    auto np = node->size;
    if (has_mixed_types) {
      feature_types.push_back(GeometryType::kMultiPoint);
      num_parts.push_back(np);
      num_rings.push_back(1);
      num_points.push_back(1);
    } else {
      num_points.push_back(np);
    }
    for (uint32_t i = 0; i < node->size; i++) {
      auto point_node = node + i + 1;
      auto point = readPoint(point_node);
      vertices.push_back(point);
      if (mbr != nullptr) {
        mbr->Expand(point.as_float());
      }
    }
  }

  void addLineString(const GeoArrowGeometryNode* node, mbr_t* mbr) {
    if (has_mixed_types) {
      feature_types.push_back(GeometryType::kLineString);
      num_parts.push_back(1);
      num_rings.push_back(1);
    }
    // push_back to num_points and vertices
    processLineString(node, mbr);
  }

  void addMultiLineString(const GeoArrowGeometryNode* node, mbr_t* mbr) {
    if (has_mixed_types) {
      feature_types.push_back(GeometryType::kMultiLineString);
      num_parts.push_back(node->size);
      num_rings.push_back(1);
    }

    for (uint32_t i = 0; i < node->size; i++) {
      auto* part_node = node + i + 1;
      // push_back to num_points and vertices
      processLineString(part_node, mbr);
    }
  }

  void addPolygon(const GeoArrowGeometryNode* node, mbr_t* mbr) {
    if (has_mixed_types) {
      feature_types.push_back(GeometryType::kPolygon);
      num_parts.push_back(1);
    }
    num_rings.push_back(node->size);
    // visit rings
    for (uint32_t i = 0; i < node->size; i++) {
      auto ring_node = node + i + 1;
      // push_back to num_points and vertices
      processLineString(ring_node, mbr);
    }
  }

  void addMultiPolygon(const GeoArrowGeometryNode* node, mbr_t* mbr) {
    if (has_mixed_types) {
      feature_types.push_back(GeometryType::kMultiPolygon);
    }
    uint32_t num_polygons = 0;
    for (auto* curr_node = node; curr_node != node + node->size; curr_node++) {
      if (node->geometry_type == GEOARROW_GEOMETRY_TYPE_POLYGON) {
        num_rings.push_back(curr_node->size);
        num_polygons++;
        // visit rings
        for (uint32_t i = 0; i < curr_node->size; i++) {
          auto ring_node = curr_node + i + 1;
          // push_back to num_points and vertices
          processLineString(ring_node, mbr);
        }
      }
    }
    num_parts.push_back(num_polygons);
  }

  void addGeometryCollection(const GeoArrowGeometryNode* root, int n_nodes, int depth,
                             mbr_t* mbr, uint32_t& ngeoms) {
    for (auto curr_node = root; curr_node != root + n_nodes; curr_node++) {
      if (curr_node->level != depth + 1) continue;
      if (curr_node->geometry_type != GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION) {
        ngeoms++;
      }
      switch (curr_node->geometry_type) {
        case GEOARROW_GEOMETRY_TYPE_POINT: {
          addPoint(curr_node, mbr);
          break;
        }
        case GEOARROW_GEOMETRY_TYPE_LINESTRING: {
          addLineString(curr_node, mbr);
          break;
        }
        case GEOARROW_GEOMETRY_TYPE_POLYGON: {
          addPolygon(curr_node, mbr);
          break;
        }
        case GEOARROW_GEOMETRY_TYPE_MULTIPOINT: {
          addMultiPoint(curr_node, mbr);
          break;
        }
        case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING: {
          addMultiLineString(curr_node, mbr);
          break;
        }
        case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON: {
          addMultiPolygon(curr_node, mbr);
          break;
        }
        case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION: {
          addGeometryCollection(curr_node, n_nodes, depth + 1, mbr, ngeoms);
          break;
        }
      }
    }
  }

  POINT_T readPoint(const GeoArrowGeometryNode* point_node) {
    bool swap_endian = (point_node->flags & GEOARROW_GEOMETRY_NODE_FLAG_SWAP_ENDIAN);
    POINT_T point;

    for (int dim = 0; dim < POINT_T::n_dim; ++dim) {
      uint64_t coord_int;
      memcpy(&coord_int, point_node->coords[dim], sizeof(uint64_t));

      if (swap_endian) {
        coord_int = __builtin_bswap64(coord_int);
      }

      double coord_double;
      memcpy(&coord_double, &coord_int, sizeof(double));

      point.set_coordinate(dim, coord_double);
    }
    return point;
  }

  void processLineString(const GeoArrowGeometryNode* node, mbr_t* mbr) {
    const uint8_t* p_coord[n_dim];
    int32_t d_coord[n_dim];

    for (int dim = 0; dim < n_dim; dim++) {
      p_coord[dim] = node->coords[dim];
      d_coord[dim] = node->coord_stride[dim];
    }

    num_points.push_back(node->size);

    for (uint32_t j = 0; j < node->size; j++) {
      POINT_T point;

      for (int dim = 0; dim < n_dim; dim++) {
        auto* coord = p_coord[dim];
        uint64_t coord_int;
        double coord_double;

        coord_int = *reinterpret_cast<const uint64_t*>(coord);
        if (node->flags & GEOARROW_GEOMETRY_NODE_FLAG_SWAP_ENDIAN) {
          coord_int = __builtin_bswap64(coord_int);
        }
        coord_double = *reinterpret_cast<double*>(&coord_int);
        point.set_coordinate(dim, coord_double);
        p_coord[dim] += d_coord[dim];
      }
      vertices.push_back(point);
      if (mbr != nullptr) {
        mbr->Expand(point.as_float());
      }
    }
  }
};

template <typename POINT_T, typename INDEX_T>
struct DeviceParsedGeometries {
  constexpr static int n_dim = POINT_T::n_dim;
  using mbr_t = Box<Point<float, n_dim> >;
  INDEX_T num_features;  // num features including nulls in Arrow table
  // will be moved to DeviceGeometries
  rmm::device_uvector<GeometryType> feature_types{0, rmm::cuda_stream_default};
  rmm::device_uvector<INDEX_T> num_geos{0, rmm::cuda_stream_default};
  // These are temp vectors during parsing
  rmm::device_uvector<INDEX_T> num_parts{0, rmm::cuda_stream_default};
  rmm::device_uvector<INDEX_T> num_rings{0, rmm::cuda_stream_default};
  rmm::device_uvector<INDEX_T> num_points{0, rmm::cuda_stream_default};
  // will be moved to DeviceGeometries
  rmm::device_uvector<POINT_T> vertices{0, rmm::cuda_stream_default};
  rmm::device_uvector<mbr_t> mbrs{0, rmm::cuda_stream_default};

  void Init(const rmm::device_async_resource_ref& mr) {
    num_features = 0;
    // Set MR of temp vectors
    num_parts = rmm::device_uvector<INDEX_T>(0, rmm::cuda_stream_default, mr);
    num_rings = rmm::device_uvector<INDEX_T>(0, rmm::cuda_stream_default, mr);
    num_points = rmm::device_uvector<INDEX_T>(0, rmm::cuda_stream_default, mr);
  }

  void Clear(rmm::cuda_stream_view stream, bool free_memory = true) {
    num_features = 0;
    feature_types.resize(0, stream);
    num_geos.resize(0, stream);
    num_parts.resize(0, stream);
    num_rings.resize(0, stream);
    num_points.resize(0, stream);
    vertices.resize(0, stream);
    if (free_memory) {
      feature_types.shrink_to_fit(stream);
      num_geos.shrink_to_fit(stream);
      num_parts.shrink_to_fit(stream);
      num_rings.shrink_to_fit(stream);
      num_points.shrink_to_fit(stream);
      vertices.shrink_to_fit(stream);
    }
  }

  GeometryType InferGeometryType(rmm::cuda_stream_view stream) const {
    rmm::device_uvector<int> d_types((int)GeometryType::kNumGeometryTypes, stream);
    auto* p_types = d_types.data();
    thrust::fill(rmm::exec_policy_nosync(stream), d_types.begin(), d_types.end(), 0);
    thrust::for_each(rmm::exec_policy_nosync(stream), feature_types.begin(),
                     feature_types.end(), [=] __device__(GeometryType type) {
                       if (type != GeometryType::kNull)
                         p_types[static_cast<int>(type)] = 1;
                     });
    std::vector<int> h_types(d_types.size());
    detail::async_copy_d2h(stream, d_types.data(), h_types.data(), d_types.size());
    stream.synchronize();
    std::unordered_set<GeometryType> unique_types(h_types.begin(), h_types.end());
    GeometryType final_type;

    switch (unique_types.size()) {
      case 0:
        final_type = GeometryType::kNull;
        break;
      case 1:
        final_type = *unique_types.begin();
        break;
      case 2: {
        if (unique_types.count(GeometryType::kPoint) &&
            unique_types.count(GeometryType::kMultiPoint)) {
          final_type = GeometryType::kMultiPoint;
        } else if (unique_types.count(GeometryType::kLineString) &&
                   unique_types.count(GeometryType::kMultiLineString)) {
          final_type = GeometryType::kMultiLineString;
        } else if (unique_types.count(GeometryType::kPolygon) &&
                   unique_types.count(GeometryType::kMultiPolygon)) {
          final_type = GeometryType::kMultiPolygon;
        } else {
          final_type = GeometryType::kGeometryCollection;
        }
      }
      default:
        final_type = GeometryType::kGeometryCollection;
    }
    return final_type;
  }
};
}  // namespace detail

template <typename POINT_T, typename INDEX_T>
class ParallelWkbLoader {
  constexpr static int n_dim = POINT_T::n_dim;
  // using low precision for memory saving
  using mbr_t = Box<Point<float, n_dim> >;

 public:
  struct Config {
    // How many threads to use for parsing WKBs
    int parallelism = 1;
    // How many rows of WKBs to process in one chunk
    // This value affects the peak memory usage and overheads
    int chunk_size = 256 * 1024;
    // Whether to allow temporary memory spilling to host memory during parsing
    // Enabling this allows to process larger datasets with limited GPU memory but will
    // reduce parsing performance
    bool temporary_memory_spilling = false;
  };

  void Init(const Config& config) {
    ArrowArrayViewInitFromType(&array_view_, NANOARROW_TYPE_BINARY);
    config_ = config;
    auto managed_mr = std::make_shared<rmm::mr::managed_memory_resource>();
    auto default_mr = rmm::mr::get_current_device_resource_ref();
    auto mr = config_.temporary_memory_spilling ? *managed_mr : default_mr;

    geoms_.Init(mr);
  }

  void Clear(rmm::cuda_stream_view stream) { geoms_.Clear(stream); }

  void Parse(rmm::cuda_stream_view stream, const ArrowArray* array, int64_t offset,
             int64_t length) {
    ArrowError arrow_error;
    if (ArrowArrayViewSetArray(&array_view_, array, &arrow_error) != NANOARROW_OK) {
      throw std::runtime_error("ArrowArrayViewSetArray error " +
                               std::string(arrow_error.message));
    }
    auto parallelism = config_.parallelism;
    auto chunk_size = config_.chunk_size;
    auto n_chunks = (length + chunk_size - 1) / chunk_size;

    bool has_mixed_types = false;
    bool has_geometry_collection = false;
    bool create_mbr = false;
    // TODO : Pre-scan to check mixed types and GeometryCollection

    // reserve space
    geoms_.num_features = length;
    geoms_.num_parts.reserve(length, stream);
    geoms_.vertices.reserve(estimateNumPoints(array, offset, length), stream);
    geoms_.mbrs.reserve(array->length, stream);

    // Batch processing to reduce the peak memory usage
    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
      auto chunk_start = chunk * chunk_size;
      auto chunk_end = std::min(length, (chunk + 1) * chunk_size);
      auto work_size = chunk_end - chunk_start;

      std::vector<std::thread> workers;

      auto thread_work_size = (work_size + parallelism - 1) / parallelism;
      std::atomic<int> next_thread_id_to_run{0};
      std::mutex mtx;
      std::condition_variable cv;
      std::unique_lock lock(mtx);

      // Each thread will parse in parallel and store results sequentially
      for (int thread_idx = 0; thread_idx < parallelism; thread_idx++) {
        auto run = [&](int tid) {
          // FIXME: SetDevice
          auto thread_work_start = chunk_start + tid * thread_work_size;
          auto thread_work_end =
              std::min(chunk_end, thread_work_start + thread_work_size);
          detail::HostParsedGeometries<POINT_T, INDEX_T> local_geoms(
              has_mixed_types, has_geometry_collection, create_mbr);
          GeoArrowWKBReader reader;
          GeoArrowError error;
          GEOARROW_THROW_NOT_OK(nullptr, GeoArrowWKBReaderInit(&reader));

          for (uint32_t work_offset = thread_work_start, i = 0;
               work_offset < thread_work_end; work_offset++, i++) {
            auto arrow_offset = work_offset + offset;
            // handle null value
            if (ArrowArrayViewIsNull(&array_view_, arrow_offset)) {
              local_geoms.AddGeometry(nullptr);
            } else {
              auto item = ArrowArrayViewGetBytesUnsafe(&array_view_, arrow_offset);
              GeoArrowGeometryView geom;

              GEOARROW_THROW_NOT_OK(
                  &error,
                  GeoArrowWKBReaderRead(&reader, {item.data.as_uint8, item.size_bytes},
                                        &geom, &error));
              local_geoms.AddGeometry(&geom);
            }
          }

          // Wait until the shared counter matches this thread's ID
          cv.wait(lock, [&]() { return tid == next_thread_id_to_run; });

          appendVector(stream, geoms_.feature_types, local_geoms.feature_types);
          appendVector(stream, geoms_.num_parts, local_geoms.num_parts);
          appendVector(stream, geoms_.num_rings, local_geoms.num_rings);
          appendVector(stream, geoms_.num_points, local_geoms.num_points);
          appendVector(stream, geoms_.vertices, local_geoms.vertices);
          appendVector(stream, geoms_.mbrs, local_geoms.mbrs);
          stream.synchronize();  // Ensure all appends are done before signaling next
          // thread
          // Signal the Next Thread
          next_thread_id_to_run++;
        };
        run(thread_idx);
        // workers.emplace_back(run, thread_idx);
      }
    }
  }

  DeviceGeometries<POINT_T, INDEX_T> Finish(rmm::cuda_stream_view stream) {
    // Calculate one by one to reduce peak memory
    // TODO: finsh loop over layers
    auto n_features = geoms_.num_features;
    rmm::device_uvector<INDEX_T> ps_num_parts(0, stream);
    calcPrefixSum(stream, geoms_.num_parts, ps_num_parts);

    rmm::device_uvector<INDEX_T> ps_num_rings(0, stream);
    calcPrefixSum(stream, geoms_.num_rings, ps_num_rings);

    rmm::device_uvector<INDEX_T> ps_num_points(0, stream);
    calcPrefixSum(stream, geoms_.num_points, ps_num_points);

    DeviceGeometries<POINT_T, INDEX_T> device_geometries;
    device_geometries.num_features = n_features;
    device_geometries.type_ = geoms_.InferGeometryType(stream);
    device_geometries.points_ = std::move(geoms_.vertices);
    device_geometries.mbrs_ = std::move(geoms_.mbrs);
    return std::move(device_geometries);
  }

 private:
  Config config_;
  ArrowArrayView array_view_;
  detail::DeviceParsedGeometries<POINT_T, INDEX_T> geoms_;

  template <typename T>
  void appendVector(rmm::cuda_stream_view stream, rmm::device_uvector<T>& d_vec,
                    const std::vector<T>& h_vec) {
    auto prev_size = d_vec.size();
    d_vec.resize(prev_size + h_vec.size(), stream);
    detail::async_copy_h2d(stream, h_vec.data(), d_vec.data() + prev_size, h_vec.size());
  }

  template <typename T>
  void calcPrefixSum(rmm::cuda_stream_view stream, rmm::device_uvector<T>& nums,
                     rmm::device_uvector<T>& ps) {
    ps.resize(nums.size() + 1, stream);
    ps.set_element_to_zero_async(0, stream);
    thrust::inclusive_scan(rmm::exec_policy_nosync(stream), nums.begin(), nums.end(),
                           ps.begin() + 1);
    nums.resize(0, stream);
    nums.shrink_to_fit(stream);
  }

  size_t estimateNumPoints(const ArrowArray* array, int64_t offset, int64_t length) {
    ArrowError arrow_error;
    if (ArrowArrayViewSetArray(&array_view_, array, &arrow_error) != NANOARROW_OK) {
      throw std::runtime_error("ArrowArrayViewSetArray error " +
                               std::string(arrow_error.message));
    }
    size_t total_bytes = 0;
    for (int64_t i = 0; i < length; i++) {
      if (!ArrowArrayViewIsNull(&array_view_, offset + i)) {
        auto item = ArrowArrayViewGetBytesUnsafe(&array_view_, offset + i);
        total_bytes += item.size_bytes - 1      // byte order
                       - 2 * sizeof(uint32_t);  // type + size
      }
    }
    return total_bytes / (sizeof(double) * POINT_T::n_dim);
  }
};
}  // namespace gpuspatial
