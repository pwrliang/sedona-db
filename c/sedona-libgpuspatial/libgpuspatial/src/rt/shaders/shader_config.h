#ifndef GPUSPATIAL_INDEX_SHADERS_SHADER_CONFIG_H
#define GPUSPATIAL_INDEX_SHADERS_SHADER_CONFIG_H

#include "gpuspatial/geom/point.cuh"

namespace gpuspatial {

#define SHADER_FUNCTION_SUFFIX "gpuspatial"
// TODO: Set separated parameters
#define SHADER_NUM_PAYLOADS (8)

#if defined(SHADER_POINT_FLOAT_2D)
using ShaderPointType = Point<float, 2>;
#define SHADER_POINT_TYPE_ID "SHADER_POINT_FLOAT_2D"
#elif defined(SHADER_POINT_DOUBLE_2D)
using ShaderPointType = Point<double, 2>;
#define SHADER_POINT_TYPE_ID "SHADER_POINT_DOUBLE_2D"
#endif

#if defined(SHADER_INDEX_UINT32)
using ShaderIndexType = uint32_t;
#define SHADER_INDEX_TYPE_ID "SHADER_INDEX_UINT32"
#elif defined(SHADER_INDEX_UINT64)
using ShaderIndexType = uint64_t;
#define SHADER_INDEX_TYPE_ID "SHADER_INDEX_UINT64"
#endif


}  // namespace gpuspatial

#endif  // GPUSPATIAL_INDEX_SHADERS_SHADER_CONFIG_H