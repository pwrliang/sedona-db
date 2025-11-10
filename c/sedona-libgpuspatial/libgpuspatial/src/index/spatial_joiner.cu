#include "gpuspatial/index/detail/launch_parameters.h"
#include "gpuspatial/index/relate_engine.cuh"
#include "gpuspatial/index/spatial_joiner.cuh"
#include "gpuspatial/utils/markers.hpp"
#include "gpuspatial/utils/stopwatch.h"
#include "gpuspatial/loader/parallel_wkb_loader.h"
#include "shaders/shader_id.hpp"

#include <rmm/exec_policy.hpp>

#define OPTIX_MAX_RAYS (1lu << 30)
namespace gpuspatial {

void SpatialJoiner::Init(const Config* config) {
  config_ = *dynamic_cast<const SpatialJoinerConfig*>(config);
  details::RTConfig rt_config = details::get_default_rt_config(config_.ptx_root);
  rt_engine_.Init(rt_config);

  stream_pool_ = std::make_unique<rmm::cuda_stream_pool>(config_.concurrency);
  ctx_pool_ = ObjectPool<SpatialJoinerContext>::create(config_.concurrency);
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, config_.stack_size_bytes));
  Clear();
}

void SpatialJoiner::Clear() {
  bvh_buffer_ = nullptr;
  // segments_.clear();
  build_type_ = GeometryType::kNumGeometryTypes;
  geometry_grouper_.clear();
  build_geometries_ = nullptr;
}

void SpatialJoiner::PushBuild(const ArrowSchema* schema, const ArrowArray* array,
                              int64_t offset, int64_t length) {
  IntervalRangeMarker marker(array->length, "PushBuild");

  // auto build_type = build_wkb_loader_.FetchGeometryType(array, offset, length);
  // is compatible type
  // if (build_type_ == GeometryType::kNumGeometryTypes) {
  //   build_type_ = build_type;
  // } else if (build_type_ != build_type) {
  //   build_type_ = GetCompatibleGeometryType(build_type_, build_type);
  // }
#if 0
  switch (build_type_) {
    case GeometryType::kPoint: {
      auto seg = std::make_shared<PointSegment<point_t>>();
      // build_wkb_loader_.Load(array, 0, array->length, *seg);
      segments_.push_back(seg);
      break;
    }
    case GeometryType::kMultiPoint: {
      auto seg = std::make_shared<MultiPointSegment<point_t, index_t>>();
      // build_wkb_loader_.Load(array, 0, array->length, *seg);
      segments_.push_back(seg);
      break;
    }
    case GeometryType::kLineString: {
      auto seg = std::make_shared<LineStringSegment<point_t, index_t>>();
      // build_wkb_loader_.Load(array, 0, array->length, *seg);
      segments_.push_back(seg);
      break;
      break;
    }
    case GeometryType::kMultiLineString: {
      auto seg = std::make_shared<MultiLineStringSegment<point_t, index_t>>();
      // build_wkb_loader_.Load(array, 0, array->length, *seg);
      segments_.push_back(seg);
      break;
    }
    case GeometryType::kPolygon: {
      auto seg = std::make_shared<PolygonSegment<point_t, index_t>>();
      // build_wkb_loader_.Load(array, 0, array->length, *seg);
      segments_.push_back(seg);
      break;
    }
    case GeometryType::kMultiPolygon: {
      auto seg = std::make_shared<MultiPolygonSegment<point_t, index_t>>();
      // build_wkb_loader_.Load(array, 0, array->length, *seg);
      segments_.push_back(seg);
      break;
    }
    default:
      throw std::runtime_error("Unimplemented build type: " +
                               GeometryTypeToString(build_type_));
  }
#endif
}

void SpatialJoiner::FinishBuilding() {
  RangeMarker marker(true, "FinishBuilding");
  auto stream = rmm::cuda_stream_default;
#if 0
  switch (build_type_) {
    case GeometryType::kPoint: {
      build_geometries_ =
          PointSegment<point_t>::LoadOnDevice<uint32_t>(stream, segments_);
      break;
    }
    case GeometryType::kMultiPoint: {
      build_geometries_ =
          MultiPointSegment<point_t, index_t>::LoadOnDevice(stream, segments_);
      break;
    }
    case GeometryType::kLineString: {
      build_geometries_ =
          LineStringSegment<point_t, index_t>::LoadOnDevice(stream, segments_);
      break;
    }
    case GeometryType::kMultiLineString: {
      build_geometries_ =
          MultiLineStringSegment<point_t, index_t>::LoadOnDevice(stream, segments_);
      break;
    }
    case GeometryType::kPolygon: {
      build_geometries_ =
          PolygonSegment<point_t, index_t>::LoadOnDevice(stream, segments_);
      break;
    }
    case GeometryType::kMultiPolygon: {
      build_geometries_ =
          MultiPolygonSegment<point_t, index_t>::LoadOnDevice(stream, segments_);
      break;
    }
    default:
      throw std::runtime_error("Unimplemented build type: " +
                               GeometryTypeToString(build_type_));
  }
  geometry_grouper_.Group(stream, *build_geometries_, config_.n_geoms_per_aabb);
  // TODO: The BVH for build is not necessary if build is point and stream is not point
  handle_ = buildBVH(stream, geometry_grouper_.get_aabbs(), bvh_buffer_);
  relate_engine_ = RelateEngine(build_geometries_.get(), &rt_engine_);
  RelateEngine<point_t, index_t>::Config re_config;
  re_config.memory_quota = config_.relate_engine_memory_quota;
  re_config.bvh_fast_build = config_.prefer_fast_build;
  re_config.bvh_fast_compact = config_.compact;
  relate_engine_.set_config(re_config);
#endif

}

void SpatialJoiner::PushStream(Context* base_ctx, const ArrowSchema* schema,
                               const ArrowArray* array, int64_t offset, int64_t length,
                               Predicate predicate, std::vector<uint32_t>* build_indices,
                               std::vector<uint32_t>* stream_indices,
                               int32_t array_index_offset) {
#if 0
  IntervalRangeMarker marker(length, "PushStream");

  auto* ctx = (SpatialJoinerContext*)base_ctx;
  ctx->cuda_stream = stream_pool_->get_stream();

  if (ctx->stream_wkb_loader == nullptr) {
    ctx->stream_wkb_loader = std::make_unique<WKBLoader<point_t>>();
  }

#ifdef GPUSPATIAL_PROFILING
  Stopwatch sw;
  sw.start();
#endif
  ctx->array_index_offset = array_index_offset + offset;

  auto stream_type = ctx->stream_wkb_loader->FetchGeometryType(array, offset, length);

  if (ctx->stream_seg == nullptr || ctx->stream_seg->GetType() != stream_type) {
    ctx->stream_seg = CreateGeometrySegment<point_t, index_t>(stream_type);
  }

  switch (stream_type) {
    case GeometryType::kPoint: {
      auto seg = std::dynamic_pointer_cast<PointSegment<point_t>>(ctx->stream_seg);
      ctx->stream_wkb_loader->Load(array, offset, length, *seg);
      ctx->stream_geometries =
          PointSegment<point_t>::LoadOnDevice<index_t>(ctx->cuda_stream, {seg});
      break;
    }
    case GeometryType::kMultiPoint: {
      auto seg =
          std::dynamic_pointer_cast<MultiPointSegment<point_t, index_t>>(ctx->stream_seg);
      ctx->stream_wkb_loader->Load(array, offset, length, *seg);
      ctx->stream_geometries =
          MultiPointSegment<point_t, index_t>::LoadOnDevice(ctx->cuda_stream, {seg});
      break;
    }
    case GeometryType::kLineString: {
      auto seg =
          std::dynamic_pointer_cast<LineStringSegment<point_t, index_t>>(ctx->stream_seg);
      ctx->stream_wkb_loader->Load(array, offset, length, *seg);
      ctx->stream_geometries =
          LineStringSegment<point_t, index_t>::LoadOnDevice(ctx->cuda_stream, {seg});
      break;
    }
    case GeometryType::kMultiLineString: {
      auto seg = std::dynamic_pointer_cast<MultiLineStringSegment<point_t, index_t>>(
          ctx->stream_seg);
      ctx->stream_wkb_loader->Load(array, offset, length, *seg);
      ctx->stream_geometries =
          MultiLineStringSegment<point_t, index_t>::LoadOnDevice(ctx->cuda_stream, {seg});
      break;
    }
    case GeometryType::kPolygon: {
      auto seg =
          std::dynamic_pointer_cast<PolygonSegment<point_t, index_t>>(ctx->stream_seg);
      ctx->stream_wkb_loader->Load(array, offset, length, *seg);
      ctx->stream_geometries =
          PolygonSegment<point_t, index_t>::LoadOnDevice(ctx->cuda_stream, {seg});
      break;
    }
    case GeometryType::kMultiPolygon: {
      auto seg = std::dynamic_pointer_cast<MultiPolygonSegment<point_t, index_t>>(
          ctx->stream_seg);
      ctx->stream_wkb_loader->Load(array, offset, length, *seg);
      ctx->stream_geometries =
          MultiPolygonSegment<point_t, index_t>::LoadOnDevice(ctx->cuda_stream, {seg});
      break;
    }
    default: {
      throw std::runtime_error("Unimplemented stream type: " +
                               GeometryTypeToString(stream_type));
    }
  }

#ifdef GPUSPATIAL_PROFILING
  sw.stop();
  ctx->parse_ms += sw.ms();
#endif

  if (build_type_ == GeometryType::kPoint) {
    if (stream_type == GeometryType::kPoint) {
      handleBuildPointStreamPoint(ctx, predicate, build_indices, stream_indices);
    } else {
      handleBuildPointStreamBox(ctx, predicate, build_indices, stream_indices);
    }
  } else {
    if (stream_type == GeometryType::kPoint) {
      handleBuildBoxStreamPoint(ctx, predicate, build_indices, stream_indices);
    } else {
      handleBuildBoxStreamBox(ctx, predicate, build_indices, stream_indices);
    }
  }
#ifdef GPUSPATIAL_PROFILING
  printf("parse %lf, alloc %lf, filter %lf, refine %lf, copy_res %lf ms\n", ctx->parse_ms,
         ctx->alloc_ms, ctx->filter_ms, ctx->refine_ms, ctx->copy_res_ms);
#endif
#endif
}

void SpatialJoiner::handleBuildPointStreamPoint(SpatialJoinerContext* ctx,
                                                Predicate predicate,
                                                std::vector<uint32_t>* build_indices,
                                                std::vector<uint32_t>* stream_indices) {
  allocateResultBuffer(ctx);

  ctx->shader_id = GetPointQueryShaderId<point_t>();
  assert(ctx->stream_geometries->get_geometry_type() == GeometryType::kPoint);

  using launch_params_t = detail::LaunchParamsPointQuery<point_t>;
  ctx->launch_params_buffer =
      std::make_unique<rmm::device_buffer>(sizeof(launch_params_t), ctx->cuda_stream);
  ctx->h_launch_params_buffer.resize(sizeof(launch_params_t));
  auto& launch_params = *(launch_params_t*)ctx->h_launch_params_buffer.data();

  launch_params.aabbs1 = geometry_grouper_.get_aabbs();
  launch_params.prefix_sum = geometry_grouper_.get_prefix_sum();
  launch_params.reordered_indices = geometry_grouper_.get_reordered_indices();
  // launch_params.mbrs1 = ArrayView<box_t>();  // no MBRs for point
  launch_params.points2 = ctx->stream_geometries->get_points();
  launch_params.handle = handle_;
  launch_params.ids = ctx->results.DeviceObject();
  CUDA_CHECK(cudaMemcpyAsync(ctx->launch_params_buffer->data(), &launch_params,
                             sizeof(launch_params_t), cudaMemcpyHostToDevice,
                             ctx->cuda_stream));

  uint32_t dim_x = std::min(OPTIX_MAX_RAYS, ctx->stream_geometries->num_features());

  filter(ctx, dim_x);
  refine(ctx, predicate, build_indices, stream_indices);
}

void SpatialJoiner::handleBuildBoxStreamPoint(SpatialJoinerContext* ctx,
                                              Predicate predicate,
                                              std::vector<uint32_t>* build_indices,
                                              std::vector<uint32_t>* stream_indices) {
  allocateResultBuffer(ctx);

  ctx->shader_id = GetPointQueryShaderId<point_t>();
  assert(ctx->stream_geometries->get_geometry_type() == GeometryType::kPoint);

  using launch_params_t = detail::LaunchParamsPointQuery<point_t>;
  ctx->launch_params_buffer =
      std::make_unique<rmm::device_buffer>(sizeof(launch_params_t), ctx->cuda_stream);
  ctx->h_launch_params_buffer.resize(sizeof(launch_params_t));
  auto& launch_params = *(launch_params_t*)ctx->h_launch_params_buffer.data();

  launch_params.aabbs1 = geometry_grouper_.get_aabbs();
  launch_params.prefix_sum = geometry_grouper_.get_prefix_sum();
  launch_params.reordered_indices = geometry_grouper_.get_reordered_indices();
  // launch_params.mbrs1 = build_geometries_->get_mbrs();
  launch_params.points2 = ctx->stream_geometries->get_points();
  launch_params.handle = handle_;
  launch_params.ids = ctx->results.DeviceObject();
  CUDA_CHECK(cudaMemcpyAsync(ctx->launch_params_buffer->data(), &launch_params,
                             sizeof(launch_params_t), cudaMemcpyHostToDevice,
                             ctx->cuda_stream));

  uint32_t dim_x = std::min(OPTIX_MAX_RAYS, ctx->stream_geometries->num_features());

  filter(ctx, dim_x);
  refine(ctx, predicate, build_indices, stream_indices);
}

void SpatialJoiner::handleBuildPointStreamBox(SpatialJoinerContext* ctx,
                                              Predicate predicate,
                                              std::vector<uint32_t>* build_indices,
                                              std::vector<uint32_t>* stream_indices) {
  allocateResultBuffer(ctx);

  ctx->shader_id = GetPointQueryShaderId<point_t>();
  assert(build_geometries_->get_geometry_type() == GeometryType::kPoint);

  using launch_params_t = detail::LaunchParamsPointQuery<point_t>;
  ctx->launch_params_buffer =
      std::make_unique<rmm::device_buffer>(sizeof(launch_params_t), ctx->cuda_stream);
  ctx->h_launch_params_buffer.resize(sizeof(launch_params_t));
  auto& launch_params = *(launch_params_t*)ctx->h_launch_params_buffer.data();

  // ctx->stream_geometries->get_mbrs();
  GeometryGrouper<point_t, index_t> grouper;
  grouper.Group(ctx->cuda_stream, *ctx->stream_geometries, config_.n_geoms_per_aabb);

  auto handle = buildBVH(ctx->cuda_stream, grouper.get_aabbs(), ctx->bvh_buffer);

  launch_params.aabbs1 = grouper.get_aabbs();
  launch_params.prefix_sum = grouper.get_prefix_sum();
  launch_params.reordered_indices = grouper.get_reordered_indices();
  // launch_params.mbrs1 = ctx->stream_geometries->get_mbrs();
  launch_params.points2 = build_geometries_->get_points();
  launch_params.handle = handle;
  launch_params.ids = ctx->results.DeviceObject();
  CUDA_CHECK(cudaMemcpyAsync(ctx->launch_params_buffer->data(), &launch_params,
                             sizeof(launch_params_t), cudaMemcpyHostToDevice,
                             ctx->cuda_stream));

  uint32_t dim_x = std::min(OPTIX_MAX_RAYS, build_geometries_->num_features());
  // IMPORTANT: In this case, the BVH is built from stream geometries and points2 are
  // build geometries, so the result pairs are (stream_id, build_id) instead of (build_id,
  // stream_id). We need to swap the output buffers to correct this.
  filter(ctx, dim_x, true);
  refine(ctx, predicate, build_indices, stream_indices);
}

void SpatialJoiner::handleBuildBoxStreamBox(SpatialJoinerContext* ctx,
                                            Predicate predicate,
                                            std::vector<uint32_t>* build_indices,
                                            std::vector<uint32_t>* stream_indices) {
  allocateResultBuffer(ctx);

  // forward cast: cast rays from stream geometries with the BVH of build geometries
  {
    auto dim_x = std::min(OPTIX_MAX_RAYS, ctx->stream_geometries->num_features());

    prepareLaunchParamsBoxQuery(ctx, true);
    filter(ctx, dim_x);
    refine(ctx, predicate, build_indices, stream_indices);
    ctx->results.Clear(ctx->cuda_stream);  // results have been copied, reuse space
  }

  // backward cast: cast rays from the build geometries with the BVH of stream geometries
  {
    auto dim_x = std::min(OPTIX_MAX_RAYS, build_geometries_->num_features());
    auto v_mbrs = ctx->stream_geometries->get_mbrs();
    rmm::device_uvector<OptixAabb> aabbs(v_mbrs.size(), ctx->cuda_stream);

    thrust::transform(rmm::exec_policy_nosync(ctx->cuda_stream), v_mbrs.begin(),
                      v_mbrs.end(), aabbs.begin(),
                      [] __device__(const box_t& mbr) { return mbr.ToOptixAabb(); });

    // Build a BVH over the MBRs of the stream geometries
    ctx->handle =
        buildBVH(ctx->cuda_stream, ArrayView<OptixAabb>(aabbs.data(), aabbs.size()),
                 ctx->bvh_buffer);
    prepareLaunchParamsBoxQuery(ctx, false);
    filter(ctx, dim_x);
    refine(ctx, predicate, build_indices, stream_indices);
  }
}

OptixTraversableHandle SpatialJoiner::buildBVH(
    const rmm::cuda_stream_view& stream, const ArrayView<OptixAabb>& aabbs,
    std::unique_ptr<rmm::device_buffer>& buffer) {
  auto buffer_size_bytes = rt_engine_.EstimateMemoryUsageForAABB(
      aabbs.size(), config_.prefer_fast_build, config_.compact);

  if (buffer == nullptr || buffer->size() < buffer_size_bytes) {
    buffer = std::make_unique<rmm::device_buffer>(buffer_size_bytes, stream);
  }

  return rt_engine_.BuildAccelCustom(stream, aabbs, *buffer, config_.prefer_fast_build,
                                     config_.compact);
}

void SpatialJoiner::allocateResultBuffer(SpatialJoinerContext* ctx) {
#ifdef GPUSPATIAL_PROFILING
  ctx->timer.start(ctx->cuda_stream);
#endif
  int64_t avail_bytes = rmm::available_device_memory().first;
  auto stream_type = ctx->stream_geometries->get_geometry_type();
  if (stream_type != GeometryType::kPoint) {
    // need to reserve space for the BVH of stream
    auto n_aabbs = ctx->stream_geometries->get_mbrs().size();

    avail_bytes -= rt_engine_.EstimateMemoryUsageForAABB(
        n_aabbs, config_.prefer_fast_build, config_.compact);
  }

  if (avail_bytes <= 0) {
    throw std::runtime_error(
        "Not enough memory to allocate result space for spatial index");
  }

  auto reserve_bytes = ceil(avail_bytes * config_.result_buffer_memory_reserve_ratio);
  reserve_bytes = reserve_bytes / config_.concurrency + 1;
  // two uint32_t for each result pair (build index, stream index) and one uint32_t for
  // the temp storage
  uint32_t n_items = reserve_bytes / (2 * sizeof(uint32_t) + sizeof(uint32_t));

  ctx->results.Init(ctx->cuda_stream, n_items);
  ctx->results.Clear(ctx->cuda_stream);
#ifdef GPUSPATIAL_PROFILING
  ctx->alloc_ms += ctx->timer.stop(ctx->cuda_stream);
#endif
}

void SpatialJoiner::prepareLaunchParamsBoxQuery(SpatialJoinerContext* ctx, bool foward) {
  using launch_params_t = detail::LaunchParamsBoxQuery<point_t>;
  ctx->launch_params_buffer =
      std::make_unique<rmm::device_buffer>(sizeof(launch_params_t), ctx->cuda_stream);
  ctx->h_launch_params_buffer.resize(sizeof(launch_params_t));
  auto& launch_params = *(launch_params_t*)ctx->h_launch_params_buffer.data();

  assert(ctx->stream_geometries->get_geometry_type() != GeometryType::kPoint);

  launch_params.aabbs1 = geometry_grouper_.get_aabbs();
  launch_params.prefix_sum = geometry_grouper_.get_prefix_sum();
  launch_params.reordered_indices = geometry_grouper_.get_reordered_indices();
  // launch_params.mbrs1 = build_geometries_->get_mbrs();
  // launch_params.mbrs2 = ctx->stream_geometries->get_mbrs();
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

void SpatialJoiner::filter(SpatialJoinerContext* ctx, uint32_t dim_x, bool swap_id) {
#ifdef GPUSPATIAL_PROFILING
  ctx->timer.start(ctx->cuda_stream);
#endif
  Stopwatch sw;

  sw.start();
  if (dim_x > 0) {
    rt_engine_.Render(ctx->cuda_stream, ctx->shader_id, dim3{dim_x, 1, 1},
                      ArrayView<char>((char*)ctx->launch_params_buffer->data(),
                                      ctx->launch_params_buffer->size()));
  }
  auto result_size = ctx->results.size(ctx->cuda_stream);
  sw.stop();

  if (swap_id && result_size > 0) {
    // swap the pair (build_id, stream_id) to (stream_id, build_id)
    thrust::transform(rmm::exec_policy_nosync(ctx->cuda_stream), ctx->results.data(),
                      ctx->results.data() + result_size, ctx->results.data(),
                      [] __device__(const thrust::pair<uint32_t, uint32_t>& pair)
                          -> thrust::pair<uint32_t, uint32_t> {
                        return thrust::make_pair(pair.second, pair.first);
                      });
  }

#ifdef GPUSPATIAL_PROFILING
  ctx->filter_ms += ctx->timer.stop(ctx->cuda_stream);
#endif
}

void SpatialJoiner::refine(SpatialJoinerContext* ctx, Predicate predicate,
                           std::vector<uint32_t>* build_indices,
                           std::vector<uint32_t>* stream_indices) {
#ifdef GPUSPATIAL_PROFILING
  ctx->timer.start(ctx->cuda_stream);
#endif
  relate_engine_.Evaluate(ctx->cuda_stream, *ctx->stream_geometries, predicate,
                          ctx->results);
#ifdef GPUSPATIAL_PROFILING
  ctx->refine_ms += ctx->timer.stop(ctx->cuda_stream);
#endif
  auto n_results = ctx->results.size(ctx->cuda_stream);

#ifdef GPUSPATIAL_PROFILING
  ctx->timer.start(ctx->cuda_stream);
#endif
  ctx->tmp_result_buffer =
      std::make_unique<rmm::device_uvector<uint32_t>>(n_results, ctx->cuda_stream);

  auto prev_size = build_indices->size();
  printf(
      "DEBUG refine(): prev build_indices size=%lu, n_results=%u, array_index_offset=%u\n",
      prev_size, n_results, ctx->array_index_offset);

  thrust::transform(
      rmm::exec_policy_nosync(ctx->cuda_stream), ctx->results.data(),
      ctx->results.data() + n_results, ctx->tmp_result_buffer->begin(),
      [] __device__(const thrust::pair<uint32_t, uint32_t>& pair) -> uint32_t {
        return pair.first;
      });

  build_indices->resize(build_indices->size() + n_results);

  CUDA_CHECK(cudaMemcpyAsync(build_indices->data() + prev_size,
                             ctx->tmp_result_buffer->data(), sizeof(uint32_t) * n_results,
                             cudaMemcpyDeviceToHost, ctx->cuda_stream));

  auto array_index_offset = ctx->array_index_offset;

  thrust::transform(
      rmm::exec_policy_nosync(ctx->cuda_stream), ctx->results.data(),
      ctx->results.data() + n_results, ctx->tmp_result_buffer->begin(),
      [=] __device__(const thrust::pair<uint32_t, uint32_t>& pair) -> uint32_t {
        return pair.second + array_index_offset;
      });

  stream_indices->resize(stream_indices->size() + n_results);
  printf("DEBUG refine(): final build_indices size=%lu, stream_indices size=%lu\n",
         build_indices->size(), stream_indices->size());

  CUDA_CHECK(cudaMemcpyAsync(stream_indices->data() + prev_size,
                             ctx->tmp_result_buffer->data(), sizeof(uint32_t) * n_results,
                             cudaMemcpyDeviceToHost, ctx->cuda_stream));
#ifdef GPUSPATIAL_PROFILING
  ctx->copy_res_ms += ctx->timer.stop(ctx->cuda_stream);
#endif
  ctx->cuda_stream.synchronize();
}

std::unique_ptr<StreamingJoiner> CreateSpatialJoiner() {
  return std::make_unique<SpatialJoiner>();
}

void InitSpatialJoiner(StreamingJoiner* index, const char* ptx_root,
                       uint32_t concurrency) {
  SpatialJoiner::SpatialJoinerConfig config;
  config.ptx_root = ptx_root;
  config.concurrency = concurrency;
  index->Init(&config);
}

}  // namespace gpuspatial
