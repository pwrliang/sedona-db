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
#include "gpuspatial/relate/relate_engine.cuh"
#include "gpuspatial/rt/launch_parameters.cuh"
#include "gpuspatial/utils/array_view.hpp"
#include "gpuspatial/utils/exception.hpp"
#include "gpuspatial/utils/helpers.cuh"
#include "gpuspatial/utils/logger.hpp"
#include "gpuspatial/mem/memory_manager.hpp"
#include "rt/shaders/shader_id.hpp"
#include "relate/relate_predicate.cuh"

#include "rmm/cuda_stream_view.hpp"
#include "rmm/device_buffer.hpp"
#include "rmm/device_uvector.hpp"
#include "rmm/exec_policy.hpp"

#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>
#include <cuda/std/iterator>
#include <algorithm>
#include <limits>

namespace gpuspatial {

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const PointArrayView<POINT_T, INDEX_T>& geom_array1,
    const PolygonArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateImpl(stream, geom_array1, MultiPointArrayView<POINT_T, INDEX_T>(), geom_array2,
               predicate, ids1, ids2, false /*inverse IM*/);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const MultiPointArrayView<POINT_T, INDEX_T>& geom_array1,
    const PolygonArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateImpl(stream, PointArrayView<POINT_T, INDEX_T>(), geom_array1, geom_array2,
               predicate, ids1, ids2, false /*inverse IM*/);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const PolygonArrayView<POINT_T, INDEX_T>& geom_array1,
    const PointArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateImpl(stream, geom_array2, MultiPointArrayView<POINT_T, INDEX_T>(), geom_array1,
               predicate, ids2, ids1, true /*inverse IM*/);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const PolygonArrayView<POINT_T, INDEX_T>& geom_array1,
    const MultiPointArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateImpl(stream, PointArrayView<POINT_T, INDEX_T>(), geom_array2, geom_array1,
               predicate, ids2, ids1, true /*inverse IM*/);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const PointArrayView<POINT_T, INDEX_T>& geom_array1,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateImpl(stream, geom_array1, MultiPointArrayView<POINT_T, INDEX_T>(), geom_array2,
               predicate, ids1, ids2, false /*inverse IM*/);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const MultiPointArrayView<POINT_T, INDEX_T>& geom_array1,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateImpl(stream, PointArrayView<POINT_T, INDEX_T>(), geom_array1, geom_array2,
               predicate, ids1, ids2, false /*inverse IM*/);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& geom_array1,
    const PointArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateImpl(stream, geom_array2, MultiPointArrayView<POINT_T, INDEX_T>(), geom_array1,
               predicate, ids2, ids1, true /*inverse IM*/);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& geom_array1,
    const MultiPointArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateImpl(stream, PointArrayView<POINT_T, INDEX_T>(), geom_array2, geom_array1,
               predicate, ids2, ids1, true /*inverse IM*/);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::EvaluateImpl(
    const rmm::cuda_stream_view& stream,
    const PointArrayView<POINT_T, INDEX_T>& point_array,
    const MultiPointArrayView<POINT_T, INDEX_T>& multi_point_array,
    const PolygonArrayView<POINT_T, INDEX_T>& poly_array, Predicate predicate,
    rmm::device_uvector<INDEX_T>& point_ids, rmm::device_uvector<INDEX_T>& poly_ids,
    bool inverse) {
  // Casting short rays from each point to do precise point-in-polygon test
  // Reference: "Geng L, Lee R, Zhang X. Rayjoin: Fast and precise spatial join.
  // InProceedings of the 38th ACM International Conference on Supercomputing 2024"
  using params_t = detail::LaunchParamsPolygonPointQuery<POINT_T, INDEX_T>;
  assert(point_array.empty() || multi_point_array.empty());
  assert(point_ids.size() == poly_ids.size());
  size_t ids_size = point_ids.size();
  GPUSPATIAL_LOG_INFO(
      "Refine with ray-tracing, (multi-)point %zu, polygon %zu, predicate %s, result size "
      "%zu, inverse %d",
      !point_array.empty() ? point_array.size() : multi_point_array.size(),
      poly_array.size(), PredicateToString(predicate), ids_size, inverse);

  if (ids_size == 0) {
    return;
  }

  auto zip_begin =
      thrust::make_zip_iterator(thrust::make_tuple(point_ids.begin(), poly_ids.begin()));
  auto zip_end =
      thrust::make_zip_iterator(thrust::make_tuple(point_ids.end(), poly_ids.end()));
  auto invalid_tuple = thrust::make_tuple(std::numeric_limits<INDEX_T>::max(),
                                          std::numeric_limits<INDEX_T>::max());

  // Sort by polygon id
  thrust::sort(rmm::exec_policy_nosync(stream), zip_begin, zip_end,
               [] __device__(const thrust::tuple<INDEX_T, INDEX_T>& tu1,
                             const thrust::tuple<INDEX_T, INDEX_T>& tu2) {
                 return thrust::get<1>(tu1) < thrust::get<1>(tu2);
               });

  rmm::device_uvector<INDEX_T> uniq_poly_ids(ids_size, stream);

  thrust::copy(rmm::exec_policy_nosync(stream), poly_ids.begin(), poly_ids.end(),
               uniq_poly_ids.begin());

  // Collect uniq polygon ids to estimate total BVH memory usage
  auto uniq_poly_ids_end = thrust::unique(rmm::exec_policy_nosync(stream),
                                          uniq_poly_ids.begin(), uniq_poly_ids.end());
  uniq_poly_ids.resize(cuda::std::distance(uniq_poly_ids.begin(), uniq_poly_ids_end),
                       stream);
  uniq_poly_ids.shrink_to_fit(stream);

  auto bvh_bytes = EstimateBVHSize(stream, poly_array, ArrayView<uint32_t>(uniq_poly_ids),
                                   config_.segs_per_aabb);
  size_t avail_bytes =
      MemoryManager::instance().get_available_device_memory() * config_.memory_quota;
  auto n_batches = bvh_bytes / avail_bytes + 1;
  auto batch_size = (ids_size + n_batches - 1) / n_batches;

  GPUSPATIAL_LOG_INFO(
      "Unique polygons %zu, memory quota %zu MB, estimated BVH size %zu MB",
      uniq_poly_ids.size(), avail_bytes / (1024 * 1024), bvh_bytes / (1024 * 1024));

  for (int batch = 0; batch < n_batches; batch++) {
    auto ids_begin = batch * batch_size;
    auto ids_end = std::min(ids_begin + batch_size, ids_size);
    auto ids_size_batch = ids_end - ids_begin;

    // Extract unique polygon IDs in this batch
    uniq_poly_ids.resize(ids_size_batch, stream);
    thrust::copy(rmm::exec_policy_nosync(stream), poly_ids.begin() + ids_begin,
                 poly_ids.begin() + ids_end, uniq_poly_ids.begin());

    // poly ids are sorted
    uniq_poly_ids_end = thrust::unique(rmm::exec_policy_nosync(stream),
                                       uniq_poly_ids.begin(), uniq_poly_ids.end());

    uniq_poly_ids.resize(cuda::std::distance(uniq_poly_ids.begin(), uniq_poly_ids_end),
                         stream);
    uniq_poly_ids.shrink_to_fit(stream);

    rmm::device_uvector<int> IMs(ids_size_batch, stream);
    rmm::device_uvector<PointLocation> locations(ids_size_batch, stream);
    rmm::device_buffer bvh_buffer(0, stream);
    rmm::device_uvector<INDEX_T> aabb_poly_ids(0, stream), aabb_ring_ids(0, stream);
    rmm::device_uvector<thrust::pair<INDEX_T, INDEX_T>> aabb_vertex_offsets(0, stream);

    // aabb id -> vertex begin[polygon] + ith point in this polygon
    auto handle = BuildBVH(stream, poly_array, ArrayView<INDEX_T>(uniq_poly_ids),
                           config_.segs_per_aabb, bvh_buffer, aabb_poly_ids,
                           aabb_ring_ids, aabb_vertex_offsets);

    params_t params;

    params.points = point_array;
    params.multi_points = multi_point_array;
    params.polygons = poly_array;
    params.uniq_polygon_ids = ArrayView<INDEX_T>(uniq_poly_ids);
    params.query_point_ids = point_ids.data() + ids_begin;
    params.query_polygon_ids = poly_ids.data() + ids_begin;
    params.query_size = ids_size_batch;
    params.IMs = ArrayView<int>(IMs);
    params.handle = handle;
    params.aabb_poly_ids = ArrayView<INDEX_T>(aabb_poly_ids);
    params.aabb_ring_ids = ArrayView<INDEX_T>(aabb_ring_ids);
    params.aabb_vertex_offsets =
        ArrayView<thrust::pair<INDEX_T, INDEX_T>>(aabb_vertex_offsets);

    rmm::device_buffer params_buffer(sizeof(params_t), stream);

    CUDA_CHECK(cudaMemcpyAsync(params_buffer.data(), &params, sizeof(params_t),
                               cudaMemcpyHostToDevice, stream.value()));

    rt_engine_->Render(
        stream, GetPolygonPointQueryShaderId<POINT_T>(),
        dim3{static_cast<unsigned int>(ids_size_batch), 1, 1},
        ArrayView<char>((char*)params_buffer.data(), params_buffer.size()));

    thrust::transform(
        rmm::exec_policy_nosync(stream),
        thrust::make_zip_iterator(thrust::make_tuple(
            point_ids.begin() + ids_begin, poly_ids.begin() + ids_begin, IMs.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            point_ids.begin() + ids_end, poly_ids.begin() + ids_end, IMs.end())),
        thrust::make_zip_iterator(thrust::make_tuple(point_ids.begin() + ids_begin,
                                                     poly_ids.begin() + ids_begin)),
        [=] __device__(const thrust::tuple<INDEX_T, INDEX_T, int>& t) {
          auto res = thrust::make_tuple(thrust::get<0>(t), thrust::get<1>(t));
          auto IM = thrust::get<2>(t);

          if (inverse) {
            IM = IntersectionMatrix::Transpose(IM);
          }

          return detail::EvaluatePredicate(predicate, IM) ? res : invalid_tuple;
        });
  }
  auto end = thrust::remove_if(rmm::exec_policy_nosync(stream), zip_begin, zip_end,
                               [=] __device__(const thrust::tuple<INDEX_T, INDEX_T>& tu) {
                                 return tu == invalid_tuple;
                               });
  size_t new_size = cuda::std::distance(zip_begin, end);
  point_ids.resize(new_size, stream);
  poly_ids.resize(new_size, stream);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::EvaluateImpl(
    const rmm::cuda_stream_view& stream,
    const PointArrayView<POINT_T, INDEX_T>& point_array,
    const MultiPointArrayView<POINT_T, INDEX_T>& multi_point_array,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& multi_poly_array, Predicate predicate,
    rmm::device_uvector<INDEX_T>& point_ids, rmm::device_uvector<INDEX_T>& multi_poly_ids,
    bool inverse) {
  using params_t = detail::LaunchParamsPointMultiPolygonQuery<POINT_T, INDEX_T>;
  assert(point_array.empty() || multi_point_array.empty());
  assert(point_ids.size() == multi_poly_ids.size());
  size_t ids_size = point_ids.size();
  GPUSPATIAL_LOG_INFO(
      "Refine with ray-tracing, (multi-)point %zu, multi-polygon %zu, predicate %s, result "
      "size %zu, inverse %d",
      !point_array.empty() ? point_array.size() : multi_point_array.size(),
      multi_poly_array.size(), PredicateToString(predicate), ids_size, inverse);

  if (ids_size == 0) {
    return;
  }
  auto zip_begin = thrust::make_zip_iterator(
      thrust::make_tuple(point_ids.begin(), multi_poly_ids.begin()));
  auto zip_end = thrust::make_zip_iterator(
      thrust::make_tuple(point_ids.end(), multi_poly_ids.end()));
  auto invalid_tuple = thrust::make_tuple(std::numeric_limits<INDEX_T>::max(),
                                          std::numeric_limits<INDEX_T>::max());

  // Sort by polygon id
  thrust::sort(rmm::exec_policy_nosync(stream), zip_begin, zip_end,
               [] __device__(const thrust::tuple<INDEX_T, INDEX_T>& tu1,
                             const thrust::tuple<INDEX_T, INDEX_T>& tu2) {
                 return thrust::get<1>(tu1) < thrust::get<1>(tu2);
               });

  rmm::device_uvector<uint32_t> uniq_multi_poly_ids(ids_size, stream);

  thrust::copy(rmm::exec_policy_nosync(stream), multi_poly_ids.begin(),
               multi_poly_ids.end(), uniq_multi_poly_ids.begin());

  // Collect uniq polygon ids to estimate total BVH memory usage
  auto uniq_multi_poly_ids_end =
      thrust::unique(rmm::exec_policy_nosync(stream), uniq_multi_poly_ids.begin(),
                     uniq_multi_poly_ids.end());
  uniq_multi_poly_ids.resize(
      cuda::std::distance(uniq_multi_poly_ids.begin(), uniq_multi_poly_ids_end), stream);
  uniq_multi_poly_ids.shrink_to_fit(stream);

  auto bvh_bytes =
      EstimateBVHSize(stream, multi_poly_array, ArrayView<uint32_t>(uniq_multi_poly_ids),
                      config_.segs_per_aabb);
  size_t avail_bytes =
      MemoryManager::instance().get_available_device_memory() * config_.memory_quota;
  auto n_batches = bvh_bytes / avail_bytes + 1;
  auto batch_size = (ids_size + n_batches - 1) / n_batches;

  GPUSPATIAL_LOG_INFO(
      "Unique multi-polygons %zu, memory quota %zu MB, estimated BVH size %zu MB",
      uniq_multi_poly_ids.size(), avail_bytes / (1024 * 1024), bvh_bytes / (1024 * 1024));

  for (int batch = 0; batch < n_batches; batch++) {
    auto ids_begin = batch * batch_size;
    auto ids_end = std::min(ids_begin + batch_size, ids_size);
    auto ids_size_batch = ids_end - ids_begin;

    // Extract multi polygon IDs in this batch
    uniq_multi_poly_ids.resize(ids_size_batch, stream);

    thrust::copy(rmm::exec_policy_nosync(stream), multi_poly_ids.begin() + ids_begin,
                 multi_poly_ids.begin() + ids_end, uniq_multi_poly_ids.begin());

    // multi polygon ids have been sorted before
    uniq_multi_poly_ids_end =
        thrust::unique(rmm::exec_policy_nosync(stream), uniq_multi_poly_ids.begin(),
                       uniq_multi_poly_ids.end());
    uniq_multi_poly_ids.resize(
        cuda::std::distance(uniq_multi_poly_ids.begin(), uniq_multi_poly_ids_end), stream);
    uniq_multi_poly_ids.shrink_to_fit(stream);

    rmm::device_uvector<int> IMs(ids_size_batch, stream);
    rmm::device_buffer bvh_buffer(0, stream);
    rmm::device_uvector<INDEX_T> aabb_multi_poly_ids(0, stream), aabb_part_ids(0, stream),
        aabb_ring_ids(0, stream);
    rmm::device_uvector<thrust::pair<INDEX_T, INDEX_T>> aabb_vertex_offsets(0, stream);
    rmm::device_uvector<INDEX_T> uniq_part_begins(0, stream);

    auto handle =
        BuildBVH(stream, multi_poly_array, ArrayView<INDEX_T>(uniq_multi_poly_ids),
                 config_.segs_per_aabb, bvh_buffer, aabb_multi_poly_ids, aabb_part_ids,
                 aabb_ring_ids, aabb_vertex_offsets, uniq_part_begins);

    params_t params;

    params.points = point_array;
    params.multi_points = multi_point_array;
    params.multi_polygons = multi_poly_array;
    params.uniq_multi_polygon_ids = ArrayView<INDEX_T>(uniq_multi_poly_ids);
    params.query_point_ids = point_ids.data() + ids_begin;
    params.query_multi_polygon_ids = multi_poly_ids.data() + ids_begin;
    params.query_size = ids_size_batch;
    params.uniq_part_begins = ArrayView<INDEX_T>(uniq_part_begins);
    params.IMs = ArrayView<int>(IMs);
    params.handle = handle;
    params.aabb_multi_poly_ids = ArrayView<INDEX_T>(aabb_multi_poly_ids);
    params.aabb_part_ids = ArrayView<INDEX_T>(aabb_part_ids);
    params.aabb_ring_ids = ArrayView<INDEX_T>(aabb_ring_ids);
    params.aabb_vertex_offsets =
        ArrayView<thrust::pair<INDEX_T, INDEX_T>>(aabb_vertex_offsets);

    rmm::device_buffer params_buffer(sizeof(params_t), stream);

    CUDA_CHECK(cudaMemcpyAsync(params_buffer.data(), &params, sizeof(params_t),
                               cudaMemcpyHostToDevice, stream.value()));

    rt_engine_->Render(
        stream, GetMultiPolygonPointQueryShaderId<POINT_T>(),
        dim3{static_cast<unsigned int>(ids_size_batch), 1, 1},
        ArrayView<char>((char*)params_buffer.data(), params_buffer.size()));

    thrust::transform(
        rmm::exec_policy_nosync(stream),
        thrust::make_zip_iterator(thrust::make_tuple(point_ids.begin() + ids_begin,
                                                     multi_poly_ids.begin() + ids_begin,
                                                     IMs.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            point_ids.begin() + ids_end, multi_poly_ids.begin() + ids_end, IMs.end())),
        thrust::make_zip_iterator(thrust::make_tuple(point_ids.begin() + ids_begin,
                                                     multi_poly_ids.begin() + ids_begin)),
        [=] __device__(const thrust::tuple<INDEX_T, INDEX_T, int>& t) {
          auto res = thrust::make_tuple(thrust::get<0>(t), thrust::get<1>(t));
          auto IM = thrust::get<2>(t);

          if (inverse) {
            IM = IntersectionMatrix::Transpose(IM);
          }

          return detail::EvaluatePredicate(predicate, IM) ? res : invalid_tuple;
        });
  }
  auto end = thrust::remove_if(rmm::exec_policy_nosync(stream), zip_begin, zip_end,
                               [=] __device__(const thrust::tuple<INDEX_T, INDEX_T>& tu) {
                                 return tu == invalid_tuple;
                               });
  size_t new_size = cuda::std::distance(zip_begin, end);
  point_ids.resize(new_size, stream);
  multi_poly_ids.resize(new_size, stream);
}

namespace {

template <typename BOX_T>
DEV_HOST_INLINE bool PaddedCovers(const BOX_T& outer, const BOX_T& inner) {
  float min_x = next_float_from_double(outer.get_min().x(), -1, 4);
  float max_x = next_float_from_double(outer.get_max().x(), 1, 4);
  float min_y = next_float_from_double(outer.get_min().y(), -1, 4);
  float max_y = next_float_from_double(outer.get_max().y(), 1, 4);
  return inner.get_min().x() >= min_x && inner.get_max().x() <= max_x &&
         inner.get_min().y() >= min_y && inner.get_max().y() <= max_y;
}

template <typename GEOM1_VIEW_T, typename GEOM2_VIEW_T, typename INDEX_T>
void PreFilterByMBR(const rmm::cuda_stream_view& stream, const GEOM1_VIEW_T& geom_array1,
                    const GEOM2_VIEW_T& geom_array2, Predicate predicate,
                    rmm::device_uvector<INDEX_T>& ids1,
                    rmm::device_uvector<INDEX_T>& ids2) {
  if (predicate != Predicate::kContains && predicate != Predicate::kCovers &&
      predicate != Predicate::kWithin && predicate != Predicate::kCoveredBy &&
      predicate != Predicate::kEquals) {
    return;
  }
  auto zip_begin =
      thrust::make_zip_iterator(thrust::make_tuple(ids1.begin(), ids2.begin()));
  auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(ids1.end(), ids2.end()));
  auto end = thrust::remove_if(
      rmm::exec_policy_nosync(stream), zip_begin, zip_end,
      [=] __device__(const thrust::tuple<INDEX_T, INDEX_T>& tu) {
        const auto& g1 = geom_array1[thrust::get<0>(tu)];
        const auto& g2 = geom_array2[thrust::get<1>(tu)];
        if (g1.empty() || g2.empty()) {
          if (predicate == Predicate::kEquals) {
            return !(g1.empty() && g2.empty());
          }
          return true;
        }
        const auto& mbr1 = g1.get_mbr();
        const auto& mbr2 = g2.get_mbr();
        if (predicate == Predicate::kContains || predicate == Predicate::kCovers) {
          return !PaddedCovers(mbr1, mbr2);
        } else if (predicate == Predicate::kWithin ||
                   predicate == Predicate::kCoveredBy) {
          return !PaddedCovers(mbr2, mbr1);
        } else {
          return !PaddedCovers(mbr1, mbr2) || !PaddedCovers(mbr2, mbr1);
        }
      });
  size_t new_size = cuda::std::distance(zip_begin, end);
  ids1.resize(new_size, stream);
  ids2.resize(new_size, stream);
}

template <bool GEOM2_IS_SINGLE_POLY, typename GEOM1_VIEW_T, typename GEOM2_VIEW_T,
          typename INDEX_T>
size_t PartitionForPass2(const rmm::cuda_stream_view& stream,
                         const GEOM1_VIEW_T& geom_array1,
                         const GEOM2_VIEW_T& geom_array2, Predicate predicate,
                         rmm::device_uvector<INDEX_T>& ids1,
                         rmm::device_uvector<INDEX_T>& ids2,
                         rmm::device_uvector<int>& IMs) {
  auto zip3_begin = thrust::make_zip_iterator(
      thrust::make_tuple(ids1.begin(), ids2.begin(), IMs.begin()));
  auto zip3_end =
      thrust::make_zip_iterator(thrust::make_tuple(ids1.end(), ids2.end(), IMs.end()));

  constexpr int kAnyIntersectMask =
      IntersectionMatrix::INTER_INTER_2D | IntersectionMatrix::INTER_BOUND_2D |
      IntersectionMatrix::BOUND_INTER_2D | IntersectionMatrix::BOUND_BOUND_2D;

  auto mid = thrust::partition(
      rmm::exec_policy_nosync(stream), zip3_begin, zip3_end,
      [=] __device__(const thrust::tuple<INDEX_T, INDEX_T, int>& t) {
        int im1 = thrust::get<2>(t);
        const auto& g1 = geom_array1[thrust::get<0>(t)];
        const auto& g2 = geom_array2[thrust::get<1>(t)];
        if (g1.empty() || g2.empty()) {
          return false;
        }
        if (!g1.get_mbr().intersects(g2.get_mbr())) {
          return false;
        }
        switch (predicate) {
          case Predicate::kIntersects:
          case Predicate::kDisjoint: {
            if ((im1 & kAnyIntersectMask) != 0) {
              return false;
            }
            if constexpr (GEOM2_IS_SINGLE_POLY) {
              return PaddedCovers(g1.get_mbr(), g2.get_mbr());
            }
            return true;
          }
          case Predicate::kTouches: {
            return (im1 & IntersectionMatrix::INTER_INTER_2D) == 0 &&
                   (im1 & IntersectionMatrix::BOUND_BOUND_2D) != 0;
          }
          case Predicate::kWithin:
          case Predicate::kCoveredBy:
          case Predicate::kEquals: {
            return (im1 & (IntersectionMatrix::INTER_EXTER_2D |
                           IntersectionMatrix::BOUND_EXTER_2D)) == 0 &&
                   (im1 & kAnyIntersectMask) != 0;
          }
          case Predicate::kContains:
          case Predicate::kCovers: {
            return (im1 & (IntersectionMatrix::EXTER_INTER_2D |
                           IntersectionMatrix::EXTER_BOUND_2D)) == 0;
          }
          default:
            return true;
        }
      });
  return cuda::std::distance(zip3_begin, mid);
}

template <typename INDEX_T>
void FilterByPredicate(const rmm::cuda_stream_view& stream, Predicate predicate,
                       rmm::device_uvector<INDEX_T>& ids1,
                       rmm::device_uvector<INDEX_T>& ids2,
                       const rmm::device_uvector<int>& IMs) {
  auto zip3_begin = thrust::make_zip_iterator(
      thrust::make_tuple(ids1.begin(), ids2.begin(), IMs.begin()));
  auto zip3_end =
      thrust::make_zip_iterator(thrust::make_tuple(ids1.end(), ids2.end(), IMs.end()));
  auto zip2_begin =
      thrust::make_zip_iterator(thrust::make_tuple(ids1.begin(), ids2.begin()));
  auto zip2_end = thrust::make_zip_iterator(thrust::make_tuple(ids1.end(), ids2.end()));
  auto invalid_tuple = thrust::make_tuple(std::numeric_limits<INDEX_T>::max(),
                                          std::numeric_limits<INDEX_T>::max());

  thrust::transform(
      rmm::exec_policy_nosync(stream), zip3_begin, zip3_end, zip2_begin,
      [=] __device__(const thrust::tuple<INDEX_T, INDEX_T, int>& t) {
        auto res = thrust::make_tuple(thrust::get<0>(t), thrust::get<1>(t));
        auto IM = thrust::get<2>(t);
        return detail::EvaluatePredicate(predicate, IM) ? res : invalid_tuple;
      });

  auto end = thrust::remove_if(rmm::exec_policy_nosync(stream), zip2_begin, zip2_end,
                               [=] __device__(const thrust::tuple<INDEX_T, INDEX_T>& tu) {
                                 return tu == invalid_tuple;
                               });
  size_t new_size = cuda::std::distance(zip2_begin, end);
  ids1.resize(new_size, stream);
  ids2.resize(new_size, stream);
}
}  // namespace

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const PolygonArrayView<POINT_T, INDEX_T>& geom_array1,
    const PolygonArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  assert(ids1.size() == ids2.size());
  if (ids1.is_empty()) {
    return;
  }
  assert(rt_engine_ != nullptr && "OptiX RT engine is required for Polygon-Polygon relate evaluation");
  PreFilterByMBR(stream, geom_array1, geom_array2, predicate, ids1, ids2);
  if (ids1.is_empty()) {
    return;
  }
  rmm::device_uvector<int> IMs(ids1.size(), stream);
  MultiPolygonArrayView<POINT_T, INDEX_T> empty_mp;
  EvaluateImpl(stream, geom_array1, empty_mp, geom_array2, ids1, ids2, IMs,
               false /*transpose*/);
  size_t num_pass2 = PartitionForPass2<true>(stream, geom_array1, geom_array2, predicate,
                                             ids1, ids2, IMs);
  if (num_pass2 > 0) {
    if (num_pass2 == ids1.size()) {
      EvaluateImpl(stream, geom_array2, empty_mp, geom_array1, ids2, ids1, IMs,
                   true /*transpose*/);
    } else {
      rmm::device_uvector<INDEX_T> sub_ids1(num_pass2, stream);
      rmm::device_uvector<INDEX_T> sub_ids2(num_pass2, stream);
      rmm::device_uvector<int> sub_IMs(num_pass2, stream);
      thrust::copy(rmm::exec_policy_nosync(stream), ids1.begin(),
                   ids1.begin() + num_pass2, sub_ids1.begin());
      thrust::copy(rmm::exec_policy_nosync(stream), ids2.begin(),
                   ids2.begin() + num_pass2, sub_ids2.begin());
      thrust::copy(rmm::exec_policy_nosync(stream), IMs.begin(), IMs.begin() + num_pass2,
                   sub_IMs.begin());
      EvaluateImpl(stream, geom_array2, empty_mp, geom_array1, sub_ids2, sub_ids1,
                   sub_IMs, true /*transpose*/);
      thrust::copy(rmm::exec_policy_nosync(stream), sub_ids1.begin(), sub_ids1.end(),
                   ids1.begin());
      thrust::copy(rmm::exec_policy_nosync(stream), sub_ids2.begin(), sub_ids2.end(),
                   ids2.begin());
      thrust::copy(rmm::exec_policy_nosync(stream), sub_IMs.begin(), sub_IMs.end(),
                   IMs.begin());
    }
  }
  FilterByPredicate(stream, predicate, ids1, ids2, IMs);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const PolygonArrayView<POINT_T, INDEX_T>& geom_array1,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  assert(ids1.size() == ids2.size());
  if (ids1.is_empty()) {
    return;
  }
  assert(rt_engine_ != nullptr && "OptiX RT engine is required for Polygon-Polygon relate evaluation");
  PreFilterByMBR(stream, geom_array1, geom_array2, predicate, ids1, ids2);
  if (ids1.is_empty()) {
    return;
  }
  rmm::device_uvector<int> IMs(ids1.size(), stream);
  PolygonArrayView<POINT_T, INDEX_T> empty_poly;
  MultiPolygonArrayView<POINT_T, INDEX_T> empty_mp;
  EvaluateImpl(stream, geom_array1, empty_mp, geom_array2, ids1, ids2, IMs,
               false /*transpose*/);
  size_t num_pass2 = PartitionForPass2<false>(stream, geom_array1, geom_array2, predicate,
                                              ids1, ids2, IMs);
  if (num_pass2 > 0) {
    if (num_pass2 == ids1.size()) {
      EvaluateImpl(stream, empty_poly, geom_array2, geom_array1, ids2, ids1, IMs,
                   true /*transpose*/);
    } else {
      rmm::device_uvector<INDEX_T> sub_ids1(num_pass2, stream);
      rmm::device_uvector<INDEX_T> sub_ids2(num_pass2, stream);
      rmm::device_uvector<int> sub_IMs(num_pass2, stream);
      thrust::copy(rmm::exec_policy_nosync(stream), ids1.begin(),
                   ids1.begin() + num_pass2, sub_ids1.begin());
      thrust::copy(rmm::exec_policy_nosync(stream), ids2.begin(),
                   ids2.begin() + num_pass2, sub_ids2.begin());
      thrust::copy(rmm::exec_policy_nosync(stream), IMs.begin(), IMs.begin() + num_pass2,
                   sub_IMs.begin());
      EvaluateImpl(stream, empty_poly, geom_array2, geom_array1, sub_ids2, sub_ids1,
                   sub_IMs, true /*transpose*/);
      thrust::copy(rmm::exec_policy_nosync(stream), sub_ids1.begin(), sub_ids1.end(),
                   ids1.begin());
      thrust::copy(rmm::exec_policy_nosync(stream), sub_ids2.begin(), sub_ids2.end(),
                   ids2.begin());
      thrust::copy(rmm::exec_policy_nosync(stream), sub_IMs.begin(), sub_IMs.end(),
                   IMs.begin());
    }
  }
  FilterByPredicate(stream, predicate, ids1, ids2, IMs);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& geom_array1,
    const PolygonArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  assert(ids1.size() == ids2.size());
  if (ids1.is_empty()) {
    return;
  }
  assert(rt_engine_ != nullptr && "OptiX RT engine is required for Polygon-Polygon relate evaluation");
  PreFilterByMBR(stream, geom_array1, geom_array2, predicate, ids1, ids2);
  if (ids1.is_empty()) {
    return;
  }
  rmm::device_uvector<int> IMs(ids1.size(), stream);
  PolygonArrayView<POINT_T, INDEX_T> empty_poly;
  MultiPolygonArrayView<POINT_T, INDEX_T> empty_mp;
  EvaluateImpl(stream, empty_poly, geom_array1, geom_array2, ids1, ids2, IMs,
               false /*transpose*/);
  size_t num_pass2 = PartitionForPass2<true>(stream, geom_array1, geom_array2, predicate,
                                             ids1, ids2, IMs);
  if (num_pass2 > 0) {
    if (num_pass2 == ids1.size()) {
      EvaluateImpl(stream, geom_array2, empty_mp, geom_array1, ids2, ids1, IMs,
                   true /*transpose*/);
    } else {
      rmm::device_uvector<INDEX_T> sub_ids1(num_pass2, stream);
      rmm::device_uvector<INDEX_T> sub_ids2(num_pass2, stream);
      rmm::device_uvector<int> sub_IMs(num_pass2, stream);
      thrust::copy(rmm::exec_policy_nosync(stream), ids1.begin(),
                   ids1.begin() + num_pass2, sub_ids1.begin());
      thrust::copy(rmm::exec_policy_nosync(stream), ids2.begin(),
                   ids2.begin() + num_pass2, sub_ids2.begin());
      thrust::copy(rmm::exec_policy_nosync(stream), IMs.begin(), IMs.begin() + num_pass2,
                   sub_IMs.begin());
      EvaluateImpl(stream, geom_array2, empty_mp, geom_array1, sub_ids2, sub_ids1,
                   sub_IMs, true /*transpose*/);
      thrust::copy(rmm::exec_policy_nosync(stream), sub_ids1.begin(), sub_ids1.end(),
                   ids1.begin());
      thrust::copy(rmm::exec_policy_nosync(stream), sub_ids2.begin(), sub_ids2.end(),
                   ids2.begin());
      thrust::copy(rmm::exec_policy_nosync(stream), sub_IMs.begin(), sub_IMs.end(),
                   IMs.begin());
    }
  }
  FilterByPredicate(stream, predicate, ids1, ids2, IMs);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& geom_array1,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  assert(ids1.size() == ids2.size());
  if (ids1.is_empty()) {
    return;
  }
  assert(rt_engine_ != nullptr && "OptiX RT engine is required for Polygon-Polygon relate evaluation");
  PreFilterByMBR(stream, geom_array1, geom_array2, predicate, ids1, ids2);
  if (ids1.is_empty()) {
    return;
  }
  rmm::device_uvector<int> IMs(ids1.size(), stream);
  PolygonArrayView<POINT_T, INDEX_T> empty_poly;
  EvaluateImpl(stream, empty_poly, geom_array1, geom_array2, ids1, ids2, IMs,
               false /*transpose*/);
  size_t num_pass2 = PartitionForPass2<false>(stream, geom_array1, geom_array2, predicate,
                                              ids1, ids2, IMs);
  if (num_pass2 > 0) {
    if (num_pass2 == ids1.size()) {
      EvaluateImpl(stream, empty_poly, geom_array2, geom_array1, ids2, ids1, IMs,
                   true /*transpose*/);
    } else {
      rmm::device_uvector<INDEX_T> sub_ids1(num_pass2, stream);
      rmm::device_uvector<INDEX_T> sub_ids2(num_pass2, stream);
      rmm::device_uvector<int> sub_IMs(num_pass2, stream);
      thrust::copy(rmm::exec_policy_nosync(stream), ids1.begin(),
                   ids1.begin() + num_pass2, sub_ids1.begin());
      thrust::copy(rmm::exec_policy_nosync(stream), ids2.begin(),
                   ids2.begin() + num_pass2, sub_ids2.begin());
      thrust::copy(rmm::exec_policy_nosync(stream), IMs.begin(), IMs.begin() + num_pass2,
                   sub_IMs.begin());
      EvaluateImpl(stream, empty_poly, geom_array2, geom_array1, sub_ids2, sub_ids1,
                   sub_IMs, true /*transpose*/);
      thrust::copy(rmm::exec_policy_nosync(stream), sub_ids1.begin(), sub_ids1.end(),
                   ids1.begin());
      thrust::copy(rmm::exec_policy_nosync(stream), sub_ids2.begin(), sub_ids2.end(),
                   ids2.begin());
      thrust::copy(rmm::exec_policy_nosync(stream), sub_IMs.begin(), sub_IMs.end(),
                   IMs.begin());
    }
  }
  FilterByPredicate(stream, predicate, ids1, ids2, IMs);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::EvaluateImpl(
    const rmm::cuda_stream_view& stream,
    const PolygonArrayView<POINT_T, INDEX_T>& probe_poly_array,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& probe_multi_poly_array,
    const PolygonArrayView<POINT_T, INDEX_T>& bvh_poly_array,
    rmm::device_uvector<INDEX_T>& probe_ids, rmm::device_uvector<INDEX_T>& bvh_ids,
    rmm::device_uvector<int>& IMs, bool transpose) {
  using params_t = detail::LaunchParamsPolygonPolygonQuery<POINT_T, INDEX_T>;
  assert(probe_poly_array.empty() || probe_multi_poly_array.empty());
  assert(probe_ids.size() == bvh_ids.size());
  size_t ids_size = probe_ids.size();
  GPUSPATIAL_LOG_INFO(
      "Refine polygon-polygon pass with ray-tracing, probe %zu, bvh polygon %zu, "
      "pairs %zu, transpose %d",
      !probe_poly_array.empty() ? probe_poly_array.size() : probe_multi_poly_array.size(),
      bvh_poly_array.size(), ids_size, transpose);

  if (ids_size == 0) {
    return;
  }

  if (!transpose) {
    auto zip_begin =
        thrust::make_zip_iterator(thrust::make_tuple(probe_ids.begin(), bvh_ids.begin()));
    auto zip_end =
        thrust::make_zip_iterator(thrust::make_tuple(probe_ids.end(), bvh_ids.end()));
    thrust::sort(rmm::exec_policy_nosync(stream), zip_begin, zip_end,
                 [] __device__(const thrust::tuple<INDEX_T, INDEX_T>& tu1,
                               const thrust::tuple<INDEX_T, INDEX_T>& tu2) {
                   return thrust::get<1>(tu1) < thrust::get<1>(tu2);
                 });
  } else {
    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(probe_ids.begin(), bvh_ids.begin(), IMs.begin()));
    auto zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(probe_ids.end(), bvh_ids.end(), IMs.end()));
    thrust::sort(rmm::exec_policy_nosync(stream), zip_begin, zip_end,
                 [] __device__(const thrust::tuple<INDEX_T, INDEX_T, int>& tu1,
                               const thrust::tuple<INDEX_T, INDEX_T, int>& tu2) {
                   return thrust::get<1>(tu1) < thrust::get<1>(tu2);
                 });
  }

  rmm::device_uvector<INDEX_T> uniq_bvh_ids(ids_size, stream);
  thrust::copy(rmm::exec_policy_nosync(stream), bvh_ids.begin(), bvh_ids.end(),
               uniq_bvh_ids.begin());
  auto uniq_bvh_ids_end = thrust::unique(rmm::exec_policy_nosync(stream),
                                         uniq_bvh_ids.begin(), uniq_bvh_ids.end());
  uniq_bvh_ids.resize(cuda::std::distance(uniq_bvh_ids.begin(), uniq_bvh_ids_end),
                      stream);
  uniq_bvh_ids.shrink_to_fit(stream);

  auto bvh_bytes = EstimateBVHSize(stream, bvh_poly_array,
                                   ArrayView<uint32_t>(uniq_bvh_ids),
                                   config_.segs_per_aabb);
  size_t avail_bytes =
      MemoryManager::instance().get_available_device_memory() * config_.memory_quota;
  auto n_batches = bvh_bytes / avail_bytes + 1;
  auto batch_size = (ids_size + n_batches - 1) / n_batches;

  GPUSPATIAL_LOG_INFO(
      "Unique BVH polygons %zu, memory quota %zu MB, estimated BVH size %zu MB",
      uniq_bvh_ids.size(), avail_bytes / (1024 * 1024), bvh_bytes / (1024 * 1024));

  for (size_t batch = 0; batch < n_batches; batch++) {
    auto ids_begin = batch * batch_size;
    auto ids_end = std::min(ids_begin + batch_size, ids_size);
    auto ids_size_batch = ids_end - ids_begin;

    if (n_batches > 1) {
      uniq_bvh_ids.resize(ids_size_batch, stream);
      thrust::copy(rmm::exec_policy_nosync(stream), bvh_ids.begin() + ids_begin,
                   bvh_ids.begin() + ids_end, uniq_bvh_ids.begin());
      uniq_bvh_ids_end = thrust::unique(rmm::exec_policy_nosync(stream),
                                        uniq_bvh_ids.begin(), uniq_bvh_ids.end());
      uniq_bvh_ids.resize(cuda::std::distance(uniq_bvh_ids.begin(), uniq_bvh_ids_end),
                          stream);
      uniq_bvh_ids.shrink_to_fit(stream);
    }

    rmm::device_buffer bvh_buffer(0, stream);
    rmm::device_uvector<INDEX_T> aabb_poly_ids(0, stream), aabb_ring_ids(0, stream);
    rmm::device_uvector<thrust::pair<INDEX_T, INDEX_T>> aabb_vertex_offsets(0, stream);

    auto handle = BuildBVH(stream, bvh_poly_array, ArrayView<INDEX_T>(uniq_bvh_ids),
                           config_.segs_per_aabb, bvh_buffer, aabb_poly_ids,
                           aabb_ring_ids, aabb_vertex_offsets);

    params_t params;
    params.probe_polygons = probe_poly_array;
    params.probe_multi_polygons = probe_multi_poly_array;
    params.bvh_polygons = bvh_poly_array;
    params.uniq_bvh_ids = ArrayView<INDEX_T>(uniq_bvh_ids);
    params.query_probe_ids = probe_ids.data() + ids_begin;
    params.query_bvh_ids = bvh_ids.data() + ids_begin;
    params.query_size = ids_size_batch;
    params.IMs = ArrayView<int>(IMs.data() + ids_begin, ids_size_batch);
    params.handle = handle;
    params.aabb_poly_ids = ArrayView<INDEX_T>(aabb_poly_ids);
    params.aabb_ring_ids = ArrayView<INDEX_T>(aabb_ring_ids);
    params.aabb_vertex_offsets =
        ArrayView<thrust::pair<INDEX_T, INDEX_T>>(aabb_vertex_offsets);
    params.transpose = transpose;

    rmm::device_buffer params_buffer(sizeof(params_t), stream);
    CUDA_CHECK(cudaMemcpyAsync(params_buffer.data(), &params, sizeof(params_t),
                               cudaMemcpyHostToDevice, stream.value()));

    rt_engine_->Render(
        stream, GetPolygonPolygonQueryShaderId<POINT_T>(),
        dim3{static_cast<unsigned int>(ids_size_batch), 1, 1},
        ArrayView<char>((char*)params_buffer.data(), params_buffer.size()));
  }
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::EvaluateImpl(
    const rmm::cuda_stream_view& stream,
    const PolygonArrayView<POINT_T, INDEX_T>& probe_poly_array,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& probe_multi_poly_array,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& bvh_multi_poly_array,
    rmm::device_uvector<INDEX_T>& probe_ids, rmm::device_uvector<INDEX_T>& bvh_ids,
    rmm::device_uvector<int>& IMs, bool transpose) {
  using params_t = detail::LaunchParamsMultiPolygonPolygonQuery<POINT_T, INDEX_T>;
  assert(probe_poly_array.empty() || probe_multi_poly_array.empty());
  assert(probe_ids.size() == bvh_ids.size());
  size_t ids_size = probe_ids.size();
  GPUSPATIAL_LOG_INFO(
      "Refine polygon-multipolygon pass with ray-tracing, probe %zu, bvh multi-polygon "
      "%zu, pairs %zu, transpose %d",
      !probe_poly_array.empty() ? probe_poly_array.size() : probe_multi_poly_array.size(),
      bvh_multi_poly_array.size(), ids_size, transpose);

  if (ids_size == 0) {
    return;
  }

  if (!transpose) {
    auto zip_begin =
        thrust::make_zip_iterator(thrust::make_tuple(probe_ids.begin(), bvh_ids.begin()));
    auto zip_end =
        thrust::make_zip_iterator(thrust::make_tuple(probe_ids.end(), bvh_ids.end()));
    thrust::sort(rmm::exec_policy_nosync(stream), zip_begin, zip_end,
                 [] __device__(const thrust::tuple<INDEX_T, INDEX_T>& tu1,
                               const thrust::tuple<INDEX_T, INDEX_T>& tu2) {
                   return thrust::get<1>(tu1) < thrust::get<1>(tu2);
                 });
  } else {
    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(probe_ids.begin(), bvh_ids.begin(), IMs.begin()));
    auto zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(probe_ids.end(), bvh_ids.end(), IMs.end()));
    thrust::sort(rmm::exec_policy_nosync(stream), zip_begin, zip_end,
                 [] __device__(const thrust::tuple<INDEX_T, INDEX_T, int>& tu1,
                               const thrust::tuple<INDEX_T, INDEX_T, int>& tu2) {
                   return thrust::get<1>(tu1) < thrust::get<1>(tu2);
                 });
  }

  rmm::device_uvector<INDEX_T> uniq_bvh_ids(ids_size, stream);
  thrust::copy(rmm::exec_policy_nosync(stream), bvh_ids.begin(), bvh_ids.end(),
               uniq_bvh_ids.begin());
  auto uniq_bvh_ids_end = thrust::unique(rmm::exec_policy_nosync(stream),
                                         uniq_bvh_ids.begin(), uniq_bvh_ids.end());
  uniq_bvh_ids.resize(cuda::std::distance(uniq_bvh_ids.begin(), uniq_bvh_ids_end),
                      stream);
  uniq_bvh_ids.shrink_to_fit(stream);

  auto bvh_bytes = EstimateBVHSize(stream, bvh_multi_poly_array,
                                   ArrayView<uint32_t>(uniq_bvh_ids),
                                   config_.segs_per_aabb);
  size_t avail_bytes =
      MemoryManager::instance().get_available_device_memory() * config_.memory_quota;
  auto n_batches = bvh_bytes / avail_bytes + 1;
  auto batch_size = (ids_size + n_batches - 1) / n_batches;

  GPUSPATIAL_LOG_INFO(
      "Unique BVH multi-polygons %zu, memory quota %zu MB, estimated BVH size %zu MB",
      uniq_bvh_ids.size(), avail_bytes / (1024 * 1024), bvh_bytes / (1024 * 1024));

  for (size_t batch = 0; batch < n_batches; batch++) {
    auto ids_begin = batch * batch_size;
    auto ids_end = std::min(ids_begin + batch_size, ids_size);
    auto ids_size_batch = ids_end - ids_begin;

    if (n_batches > 1) {
      uniq_bvh_ids.resize(ids_size_batch, stream);
      thrust::copy(rmm::exec_policy_nosync(stream), bvh_ids.begin() + ids_begin,
                   bvh_ids.begin() + ids_end, uniq_bvh_ids.begin());
      uniq_bvh_ids_end = thrust::unique(rmm::exec_policy_nosync(stream),
                                        uniq_bvh_ids.begin(), uniq_bvh_ids.end());
      uniq_bvh_ids.resize(cuda::std::distance(uniq_bvh_ids.begin(), uniq_bvh_ids_end),
                          stream);
      uniq_bvh_ids.shrink_to_fit(stream);
    }

    rmm::device_buffer bvh_buffer(0, stream);
    rmm::device_uvector<INDEX_T> aabb_multi_poly_ids(0, stream), aabb_part_ids(0, stream),
        aabb_ring_ids(0, stream);
    rmm::device_uvector<thrust::pair<INDEX_T, INDEX_T>> aabb_vertex_offsets(0, stream);
    rmm::device_uvector<INDEX_T> uniq_part_begins(0, stream);

    auto handle =
        BuildBVH(stream, bvh_multi_poly_array, ArrayView<INDEX_T>(uniq_bvh_ids),
                 config_.segs_per_aabb, bvh_buffer, aabb_multi_poly_ids, aabb_part_ids,
                 aabb_ring_ids, aabb_vertex_offsets, uniq_part_begins);

    params_t params;
    params.probe_polygons = probe_poly_array;
    params.probe_multi_polygons = probe_multi_poly_array;
    params.bvh_multi_polygons = bvh_multi_poly_array;
    params.uniq_bvh_ids = ArrayView<INDEX_T>(uniq_bvh_ids);
    params.uniq_part_begins = ArrayView<INDEX_T>(uniq_part_begins);
    params.query_probe_ids = probe_ids.data() + ids_begin;
    params.query_bvh_ids = bvh_ids.data() + ids_begin;
    params.query_size = ids_size_batch;
    params.IMs = ArrayView<int>(IMs.data() + ids_begin, ids_size_batch);
    params.handle = handle;
    params.aabb_multi_poly_ids = ArrayView<INDEX_T>(aabb_multi_poly_ids);
    params.aabb_part_ids = ArrayView<INDEX_T>(aabb_part_ids);
    params.aabb_ring_ids = ArrayView<INDEX_T>(aabb_ring_ids);
    params.aabb_vertex_offsets =
        ArrayView<thrust::pair<INDEX_T, INDEX_T>>(aabb_vertex_offsets);
    params.transpose = transpose;

    rmm::device_buffer params_buffer(sizeof(params_t), stream);
    CUDA_CHECK(cudaMemcpyAsync(params_buffer.data(), &params, sizeof(params_t),
                               cudaMemcpyHostToDevice, stream.value()));

    rt_engine_->Render(
        stream, GetMultiPolygonPolygonQueryShaderId<POINT_T>(),
        dim3{static_cast<unsigned int>(ids_size_batch), 1, 1},
        ArrayView<char>((char*)params_buffer.data(), params_buffer.size()));
  }
}

// Explicit instantiation of RelateEngine class template
template class RelateEngine<Point<double, 2>, uint32_t>;

}  // namespace gpuspatial
