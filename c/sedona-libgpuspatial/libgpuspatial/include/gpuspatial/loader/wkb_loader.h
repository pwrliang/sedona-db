#pragma once
#include <cassert>

#include "gpuspatial/geom/box.cuh"
#include "gpuspatial/geom/geometry_type.cuh"
#include "gpuspatial/loader/device_geometries.cuh"

#include "geoarrow/geoarrow.hpp"

#include "nanoarrow/nanoarrow.h"
#include "nanoarrow/nanoarrow.hpp"
namespace gpuspatial {

template <typename POINT_T>
class WKBLoader {
  using point_t = POINT_T;
  using box_t = Box<point_t>;
  using scalar_t = typename point_t::scalar_t;
  constexpr static int n_dim = point_t::n_dim;

 public:
  WKBLoader() {
    GEOARROW_THROW_NOT_OK(nullptr, GeoArrowWKBReaderInit(&reader_));
    ArrowArrayViewInitFromType(&array_view_, NANOARROW_TYPE_BINARY);
  }

  GeometryType FetchGeometryType(const ArrowArray* array, int64_t offset, int64_t length,
                                 bool get_first = false) {
    ArrowError arrow_error;
    GeoArrowError error;
    GeometryType prev_type = GeometryType::kNumGeometryTypes;

    if (ArrowArrayViewSetArray(&array_view_, array, &arrow_error) != NANOARROW_OK) {
      throw std::runtime_error("ArrowArrayViewSetArray error " +
                               std::string(arrow_error.message));
    }

    for (int64_t i = 0; i < length; i++) {
      if (!ArrowArrayViewIsNull(&array_view_, offset + i)) {
        auto item = ArrowArrayViewGetBytesUnsafe(&array_view_, offset + i);
        auto* s = (struct WKBReaderPrivate*)reader_.private_data;

        s->data = item.data.as_uint8;
        s->data0 = s->data;
        s->size_bytes = item.size_bytes;

        NANOARROW_THROW_NOT_OK(WKBReaderReadEndian(s, &error));
        uint32_t geometry_type;
        NANOARROW_THROW_NOT_OK(WKBReaderReadUInt32(s, &geometry_type, &error));

        auto type = toGeometryType((GeoArrowGeometryType)geometry_type);
        if (prev_type == GeometryType::kNumGeometryTypes) {
          prev_type = type;
          if (get_first) {
            break;
          }
        } else if (prev_type != type) {
          prev_type = GetCompatibleGeometryType(prev_type, type);
        }
      }
    }

    return prev_type;
  }

  template <typename GEOMETRY_SEGMENT_T>
  void Load(const ArrowArray* array, int64_t offset, int64_t length,
            GEOMETRY_SEGMENT_T& seg) {
    seg.Reserve(length);
    seg.Clear();

    iterateArrowArray(array, offset, length,
                      [&](const GeoArrowGeometryView* geom) { seg.Add(geom); });
  }

  template <typename INDEX_T>
  void Load(const ArrowArray* array, int64_t offset, int64_t length,
            PolygonSegment<point_t, INDEX_T>& seg) {
    ArrowArrayView view;
    ArrowArrayViewInitFromType(&view, NANOARROW_TYPE_BINARY);
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(&view, array, nullptr));
    auto bytes = view.buffer_views[NANOARROW_MAX_FIXED_BUFFERS - 1].size_bytes;
    auto estimated_n_points = bytes / (sizeof(double) * n_dim);

    seg.Reserve(array->length);
    seg.ReservePoints(estimated_n_points);
    seg.Clear();

    // for each polygon
    iterateArrowArray(array, offset, length, [&](const GeoArrowGeometryView* geom) {
      if (geom != nullptr) {
        seg.Add(geom);
      } else {
        throw std::runtime_error("Not support adding empty polygon yet");
      }
    });
  }

  template <typename INDEX_T>
  void Load(const ArrowArray* array, int64_t offset, int64_t length,
            MultiPolygonSegment<point_t, INDEX_T>& seg) {
    nanoarrow::UniqueArrayView view;
    ArrowArrayViewInitFromType(view.get(), NANOARROW_TYPE_BINARY);
    NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(view.get(), array, nullptr));
    auto bytes = view->buffer_views[NANOARROW_MAX_FIXED_BUFFERS - 1].size_bytes;
    auto estimated_n_points = bytes / (sizeof(double) * n_dim);

    seg.Reserve(array->length);
    seg.ReservePoints(estimated_n_points);
    seg.Clear();

    // for each polygon
    iterateArrowArray(array, offset, length, [&](const GeoArrowGeometryView* geom) {
      if (geom != nullptr) {
        seg.Add(geom);
      } else {
        throw std::runtime_error("Not support adding empty polygon yet");
      }
    });
  }

 private:
  GeoArrowError error_{};
  GeoArrowWKBReader reader_;
  ArrowArrayView array_view_;

  GeometryType toGeometryType(GeoArrowGeometryType geo_arrow_type) {
    GeometryType type = GeometryType::kNumGeometryTypes;
    switch (geo_arrow_type) {
      case GEOARROW_GEOMETRY_TYPE_POINT: {
        type = GeometryType::kPoint;
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_LINESTRING: {
        type = GeometryType::kLineString;
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_POLYGON: {
        type = GeometryType::kPolygon;
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_MULTIPOINT: {
        type = GeometryType::kMultiPoint;
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING: {
        type = GeometryType::kMultiLineString;
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON: {
        type = GeometryType::kMultiPolygon;
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION: {
        type = GeometryType::kGeometryCollection;
        break;
      }
      default: {
        throw std::runtime_error("Unsupported type " +
                                 std::string(GeoArrowGeometryTypeString(geo_arrow_type)));
      }
    }
    return type;
  }

  template <typename FUNC_T>
  void iterateArrowArray(const ArrowArray* array, int64_t offset, int64_t length,
                         const FUNC_T& func) {
    ArrowError arrow_error;
    if (ArrowArrayViewSetArray(&array_view_, array, &arrow_error) != NANOARROW_OK) {
      throw std::runtime_error("ArrowArrayViewSetArray error " +
                               std::string(arrow_error.message));
    }

    for (int64_t i = 0; i < length; i++) {
      if (!ArrowArrayViewIsNull(&array_view_, offset + i)) {
        auto item = ArrowArrayViewGetBytesUnsafe(&array_view_, offset + i);
        GeoArrowGeometryView geom;

        GEOARROW_THROW_NOT_OK(
            &error_,
            GeoArrowWKBReaderRead(&reader_, {item.data.as_uint8, item.size_bytes}, &geom,
                                  &error_));
        func(&geom);
      } else {
        func(nullptr);
      }
    }
  }

  // Copied from GeoArrow, it is faster than using GeoArrowWKBReaderRead
  struct WKBReaderPrivate {
    const uint8_t* data;
    int64_t size_bytes;
    const uint8_t* data0;
    int need_swapping;
    GeoArrowGeometry geom;
  };

  int WKBReaderReadEndian(struct WKBReaderPrivate* s, struct GeoArrowError* error) {
    if (s->size_bytes > 0) {
      s->need_swapping = s->data[0] != GEOARROW_NATIVE_ENDIAN;
      s->data++;
      s->size_bytes--;
      return GEOARROW_OK;
    } else {
      GeoArrowErrorSet(error, "Expected endian byte but found end of buffer at byte %ld",
                       (long)(s->data - s->data0));
      return EINVAL;
    }
  }

  int WKBReaderReadUInt32(struct WKBReaderPrivate* s, uint32_t* out,
                          struct GeoArrowError* error) {
    if (s->size_bytes >= 4) {
      memcpy(out, s->data, sizeof(uint32_t));
      s->data += sizeof(uint32_t);
      s->size_bytes -= sizeof(uint32_t);
      if (s->need_swapping) {
        *out = bswap_32(*out);
      }
      return GEOARROW_OK;
    } else {
      GeoArrowErrorSet(error, "Expected uint32 but found end of buffer at byte %ld",
                       (long)(s->data - s->data0));
      return EINVAL;
    }
  }

  uint32_t bswap_32(uint32_t x) {
    return (((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x & 0xFF0000) >> 8) |
            ((x & 0xFF000000) >> 24));
  }
};
}  // namespace gpuspatial
