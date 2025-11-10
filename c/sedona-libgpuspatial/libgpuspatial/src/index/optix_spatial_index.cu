#include "gpuspatial/index/detail/launch_parameters.h"
#include "gpuspatial/index/optix_spatial_index.cuh"
#include "gpuspatial/index/relate_engine.cuh"
#include "gpuspatial/utils/markers.hpp"
#include "gpuspatial/utils/mem_utils.hpp"
#include "gpuspatial/utils/stopwatch.h"
#include "shaders/shader_id.hpp"

#include <rmm/exec_policy.hpp>

#define OPTIX_MAX_RAYS (1u << 30)
namespace gpuspatial {

template <typename POINT_T, typename INDEX_T>
void OptixSpatialIndex<POINT_T, INDEX_T>::Init(
    const typename SpatialIndex<POINT_T, INDEX_T>::Config* config) {
  config_ = *dynamic_cast<const SpatialIndexConfig*>(config);
  details::RTConfig rt_config = details::get_default_rt_config(config_.ptx_root);
  rt_engine_.Init(rt_config);

  stream_pool_ = std::make_unique<rmm::cuda_stream_pool>(config_.concurrency);
  ctx_pool_ = ObjectPool<SpatialIndexContext>::create(config_.concurrency);
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, config_.stack_size_bytes));
  Clear();
}

template <typename POINT_T, typename INDEX_T>
void OptixSpatialIndex<POINT_T, INDEX_T>::Clear() {
  bvh_buffer_ = nullptr;
  box_seg_ = std::make_shared<BoxSegment<point_t, index_t>>();
  geometry_grouper.clear();
  build_geometries_ = nullptr;
}

template <typename POINT_T, typename INDEX_T>
void OptixSpatialIndex<POINT_T, INDEX_T>::PushBuild(const box_t* boxes,
                                                    uint32_t n_boxes) {
  IntervalRangeMarker marker(n_boxes, "PushBuild");

  if (boxes == nullptr || n_boxes == 0) {
    return;  // nothing to do
  }
  box_seg_->Reserve(n_boxes);
  for (uint32_t i = 0; i < n_boxes; i++) {
    box_seg_->Add(&boxes[i]);
  }
}

template <typename POINT_T, typename INDEX_T>
void OptixSpatialIndex<POINT_T, INDEX_T>::FinishBuilding() {
  RangeMarker marker(true, "FinishBuilding");
  auto stream = rmm::cuda_stream_default;

  build_geometries_ =
      BoxSegment<point_t, index_t>::template LoadOnDevice(stream, {box_seg_});

  geometry_grouper.Group(stream, *build_geometries_, config_.n_geoms_per_aabb);
  handle_ = buildBVH(stream, geometry_grouper.get_aabbs(), bvh_buffer_);
}

template <typename POINT_T, typename INDEX_T>
void OptixSpatialIndex<POINT_T, INDEX_T>::PushStream(
    typename SpatialIndex<POINT_T, INDEX_T>::Context* base_ctx, const point_t* points,
    uint32_t n_points, std::vector<uint32_t>* build_indices,
    std::vector<uint32_t>* stream_indices) {
  IntervalRangeMarker marker(n_points, "PushStream");

  auto* ctx = (SpatialIndexContext*)base_ctx;
  ctx->cuda_stream = stream_pool_->get_stream();

#ifdef GPUSPATIAL_PROFILING
  Stopwatch sw;
  sw.start();
#endif

  ctx->stream_points =
      std::make_shared<rmm::device_uvector<point_t>>(n_points, ctx->cuda_stream);

  detail::async_copy_h2d(ctx->cuda_stream, points, ctx->stream_points->data(), n_points);

#ifdef GPUSPATIAL_PROFILING
  sw.stop();
  ctx->parse_ms += sw.ms();
#endif

  // Box-Point query
  allocateResultBuffer(ctx);

  prepareLaunchParamsPointQuery(ctx);

  uint32_t dim_x = std::min(OPTIX_MAX_RAYS, n_points);

  filter(ctx, dim_x, build_indices, stream_indices);
  ctx->stream_points = nullptr;
#ifdef GPUSPATIAL_PROFILING
  printf("parse %lf, alloc %lf, filter %lf, refine %lf, copy_res %lf ms\n", ctx->parse_ms,
         ctx->alloc_ms, ctx->filter_ms, ctx->refine_ms, ctx->copy_res_ms);
#endif
}

template <typename POINT_T, typename INDEX_T>
void OptixSpatialIndex<POINT_T, INDEX_T>::PushStream(
    typename SpatialIndex<POINT_T, INDEX_T>::Context* base_ctx, const box_t* boxes,
    uint32_t n_boxes, std::vector<uint32_t>* build_indices,
    std::vector<uint32_t>* stream_indices) {
  IntervalRangeMarker marker(n_boxes, "PushStream");

  auto* ctx = (SpatialIndexContext*)base_ctx;
  ctx->cuda_stream = stream_pool_->get_stream();

#ifdef GPUSPATIAL_PROFILING
  Stopwatch sw;
  sw.start();
#endif

  ctx->stream_boxes =
      std::make_shared<rmm::device_uvector<box_t>>(n_boxes, ctx->cuda_stream);

  detail::async_copy_h2d(ctx->cuda_stream, boxes, ctx->stream_boxes->data(), n_boxes);

#ifdef GPUSPATIAL_PROFILING
  sw.stop();
  ctx->parse_ms += sw.ms();
#endif

  // Box-Point query
  allocateResultBuffer(ctx);

  // forward cast: cast rays from stream geometries with the BVH of build geometries
  {
    auto dim_x = std::min(OPTIX_MAX_RAYS, n_boxes);

    prepareLaunchParamsBoxQuery(ctx, true);
    filter(ctx, dim_x, build_indices, stream_indices);
    ctx->results.Clear(ctx->cuda_stream);  // results have been copied, reuse space
  }

  // backward cast: cast rays from the build geometries with the BVH of stream geometries
  {
    auto dim_x = std::min(OPTIX_MAX_RAYS, (uint32_t)build_geometries_->get_mbrs().size());
    rmm::device_uvector<OptixAabb> aabbs(ctx->stream_boxes->size(), ctx->cuda_stream);

    thrust::transform(rmm::exec_policy_nosync(ctx->cuda_stream),
                      ctx->stream_boxes->begin(), ctx->stream_boxes->end(), aabbs.begin(),
                      [] __device__(const box_t& mbr) { return mbr.ToOptixAabb(); });

    // Build a BVH over the MBRs of the stream geometries
    ctx->handle =
        buildBVH(ctx->cuda_stream, ArrayView<OptixAabb>(aabbs.data(), aabbs.size()),
                 ctx->bvh_buffer);
    prepareLaunchParamsBoxQuery(ctx, false);
    filter(ctx, dim_x, build_indices, stream_indices);
  }
  ctx->stream_boxes = nullptr;
#ifdef GPUSPATIAL_PROFILING
  printf("parse %lf, alloc %lf, filter %lf, refine %lf, copy_res %lf ms\n", ctx->parse_ms,
         ctx->alloc_ms, ctx->filter_ms, ctx->refine_ms, ctx->copy_res_ms);
#endif
}

template <typename POINT_T, typename INDEX_T>
OptixTraversableHandle OptixSpatialIndex<POINT_T, INDEX_T>::buildBVH(
    const rmm::cuda_stream_view& stream, const ArrayView<OptixAabb>& aabbs,
    std::unique_ptr<rmm::device_buffer>& buffer) {
  if (buffer == nullptr) {
    auto buffer_size_bytes = rt_engine_.EstimateMemoryUsageForAABB(
        aabbs.size(), config_.prefer_fast_build, config_.compact);

    buffer = std::make_unique<rmm::device_buffer>(buffer_size_bytes, stream);
  }

  return rt_engine_.BuildAccelCustom(stream, aabbs, *buffer, config_.prefer_fast_build,
                                     config_.compact);
}

template <typename POINT_T, typename INDEX_T>
void OptixSpatialIndex<POINT_T, INDEX_T>::allocateResultBuffer(SpatialIndexContext* ctx) {
#ifdef GPUSPATIAL_PROFILING
  ctx->timer.start(ctx->cuda_stream);
#endif
  int64_t avail_bytes = rmm::available_device_memory().first;
  if (ctx->stream_boxes != nullptr) {
    // need to reserve space for the BVH of stream
    auto n_aabbs = ctx->stream_boxes->size();

    avail_bytes -= rt_engine_.EstimateMemoryUsageForAABB(
        n_aabbs, config_.prefer_fast_build, config_.compact);
  }

  if (avail_bytes <= 0) {
    throw std::runtime_error(
        "Not enough memory to allocate result space for spatial index");
  }

  auto reserve_bytes = ceil(avail_bytes * config_.result_buffer_memory_reserve_ratio);
  reserve_bytes = reserve_bytes / config_.concurrency + 1;
  // two uint32_t for each result pair (build index, stream index) and one int64_t for the
  // temp storage
  uint32_t n_items = reserve_bytes / (2 * sizeof(uint32_t) + sizeof(int64_t));

  ctx->results.Init(ctx->cuda_stream, n_items);
  ctx->results.Clear(ctx->cuda_stream);
#ifdef GPUSPATIAL_PROFILING
  ctx->alloc_ms += ctx->timer.stop(ctx->cuda_stream);
#endif
}

template <typename POINT_T, typename INDEX_T>
void OptixSpatialIndex<POINT_T, INDEX_T>::prepareLaunchParamsPointQuery(
    SpatialIndexContext* ctx) {
  ctx->shader_id = GetPointQueryShaderId<point_t>();
  assert(ctx->stream_points != nullptr);

  using launch_params_t = detail::LaunchParamsPointQuery<point_t>;
  ctx->launch_params_buffer =
      std::make_unique<rmm::device_buffer>(sizeof(launch_params_t), ctx->cuda_stream);
  ctx->h_launch_params_buffer.resize(sizeof(launch_params_t));
  auto& launch_params = *(launch_params_t*)ctx->h_launch_params_buffer.data();

  launch_params.aabbs1 = geometry_grouper.get_aabbs();
  launch_params.prefix_sum = geometry_grouper.get_prefix_sum();
  launch_params.reordered_indices = geometry_grouper.get_reordered_indices();
  // launch_params.mbrs1 = build_geometries_->get_mbrs();
  launch_params.points2 = ArrayView<point_t>(*ctx->stream_points);
  launch_params.handle = handle_;
  launch_params.ids = ctx->results.DeviceObject();
  CUDA_CHECK(cudaMemcpyAsync(ctx->launch_params_buffer->data(), &launch_params,
                             sizeof(launch_params_t), cudaMemcpyHostToDevice,
                             ctx->cuda_stream));
}

template <typename POINT_T, typename INDEX_T>
void OptixSpatialIndex<POINT_T, INDEX_T>::prepareLaunchParamsBoxQuery(
    SpatialIndexContext* ctx, bool foward) {
  using launch_params_t = detail::LaunchParamsBoxQuery<point_t>;
  ctx->launch_params_buffer =
      std::make_unique<rmm::device_buffer>(sizeof(launch_params_t), ctx->cuda_stream);
  ctx->h_launch_params_buffer.resize(sizeof(launch_params_t));
  auto& launch_params = *(launch_params_t*)ctx->h_launch_params_buffer.data();

  assert(ctx->stream_boxes != nullptr);

  launch_params.aabbs1 = geometry_grouper.get_aabbs();
  launch_params.prefix_sum = geometry_grouper.get_prefix_sum();
  launch_params.reordered_indices = geometry_grouper.get_reordered_indices();
  // launch_params.mbrs1 = build_geometries_->get_mbrs();
  launch_params.mbrs2 = ArrayView<box_t>(*ctx->stream_boxes);
  if (foward) {
    launch_params.handle = handle_;
    ctx->shader_id = GetBoxQueryForwardShaderId<point_t>();
  } else {
    launch_params.handle = ctx->handle;
    ctx->shader_id = GetBoxQueryBackwardShaderId<point_t>();
  }

  launch_params.ids = ctx->results.DeviceObject();
  CUDA_CHECK(cudaMemcpyAsync(ctx->launch_params_buffer->data(), &launch_params,
                             sizeof(launch_params_t), cudaMemcpyHostToDevice,
                             ctx->cuda_stream));
}

template <typename POINT_T, typename INDEX_T>
void OptixSpatialIndex<POINT_T, INDEX_T>::filter(SpatialIndexContext* ctx, uint32_t dim_x,
                                                 std::vector<uint32_t>* build_indices,
                                                 std::vector<uint32_t>* stream_indices) {
#ifdef GPUSPATIAL_PROFILING
  ctx->timer.start(ctx->cuda_stream);
#endif
  rt_engine_.Render(ctx->cuda_stream, ctx->shader_id, dim3{dim_x, 1, 1},
                    ArrayView<char>((char*)ctx->launch_params_buffer->data(),
                                    ctx->launch_params_buffer->size()));
#ifdef GPUSPATIAL_PROFILING
  ctx->filter_ms += ctx->timer.stop(ctx->cuda_stream);
#endif
  auto prev_size = build_indices->size();
  auto n_results = ctx->results.size(ctx->cuda_stream);

#ifdef GPUSPATIAL_PROFILING
  ctx->timer.start(ctx->cuda_stream);
#endif

  rmm::device_uvector<uint32_t> tmp_result_buffer(n_results, ctx->cuda_stream);

  thrust::transform(
      rmm::exec_policy_nosync(ctx->cuda_stream), ctx->results.data(),
      ctx->results.data() + n_results, tmp_result_buffer.begin(),
      [] __device__(const thrust::pair<uint32_t, uint32_t>& pair) -> uint32_t {
        return pair.first;
      });

  build_indices->resize(build_indices->size() + n_results);

  CUDA_CHECK(cudaMemcpyAsync(build_indices->data() + prev_size, tmp_result_buffer.data(),
                             sizeof(uint32_t) * n_results, cudaMemcpyDeviceToHost,
                             ctx->cuda_stream));

  thrust::transform(
      rmm::exec_policy_nosync(ctx->cuda_stream), ctx->results.data(),
      ctx->results.data() + n_results, tmp_result_buffer.begin(),
      [=] __device__(const thrust::pair<uint32_t, uint32_t>& pair) -> uint32_t {
        return pair.second;
      });

  stream_indices->resize(stream_indices->size() + n_results);

  CUDA_CHECK(cudaMemcpyAsync(stream_indices->data() + prev_size, tmp_result_buffer.data(),
                             sizeof(uint32_t) * n_results, cudaMemcpyDeviceToHost,
                             ctx->cuda_stream));
#ifdef GPUSPATIAL_PROFILING
  ctx->copy_res_ms += ctx->timer.stop(ctx->cuda_stream);
#endif
  ctx->cuda_stream.synchronize();
}

template <typename POINT_T, typename INDEX_T>
std::unique_ptr<SpatialIndex<POINT_T, INDEX_T>> CreateOptixSpatialIndex() {
  return std::make_unique<OptixSpatialIndex<POINT_T, INDEX_T>>();
}

template <typename POINT_T, typename INDEX_T>
void InitOptixSpatialIndex(SpatialIndex<POINT_T, INDEX_T>* index, const char* ptx_root,
                           uint32_t concurrency) {
  typename OptixSpatialIndex<POINT_T, INDEX_T>::SpatialIndexConfig config;
  config.ptx_root = ptx_root;
  config.concurrency = concurrency;
  index->Init(&config);
}

template class OptixSpatialIndex<Point<float, 2>, uint32_t>;
template class OptixSpatialIndex<Point<double, 2>, uint32_t>;

// template std::unique_ptr<SpatialIndex<Point<float, 2>, uint32_t>>
// CreateOptixSpatialIndex(); template void
// InitOptixSpatialIndex(SpatialIndex<Point<float, 2>, uint32_t>* index, const char*
// ptx_root,
//                                     uint32_t concurrency);

// template class SpatialIndex<double, 2>;
// template std::unique_ptr<SpatialIndex<double, 2>> CreateOptixSpatialIndex();
//
// // template class SpatialIndex<double, 2>;
// template void InitOptixSpatialIndex(SpatialIndex<double, 2>* index, const char*
// ptx_root,
//                                     uint32_t concurrency);

}  // namespace gpuspatial
