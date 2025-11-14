#pragma once
#include <memory>
#include "gpuspatial/index/streaming_joiner.hpp"

namespace gpuspatial {
std::unique_ptr<StreamingJoiner> CreateSpatialJoiner();

void InitSpatialJoiner(StreamingJoiner* index, const char* ptx_root, uint32_t concurrency);
}  // namespace gpuspatial

