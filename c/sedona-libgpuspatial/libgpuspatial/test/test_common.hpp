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
#ifndef GPUSPATIAL_TEST_TEST_COMMON_HPP
#define GPUSPATIAL_TEST_TEST_COMMON_HPP
#include "gpuspatial/geom/point.cuh"
#include "gpuspatial/utils/array_view.h"
#include "gpuspatial/utils/pinned_vector.h"

#include <gtest/gtest.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include "rmm/exec_policy.hpp"

namespace TestUtils {
using PointTypes =
    ::testing::Types<gpuspatial::Point<float, 2>, gpuspatial::Point<double, 2>>;
using PointIndexTypePairs =
    ::testing::Types<std::pair<gpuspatial::Point<float, 2>, uint32_t>,
                     std::pair<gpuspatial::Point<double, 2>, uint32_t>,
                     std::pair<gpuspatial::Point<float, 2>, uint64_t>,
                     std::pair<gpuspatial::Point<double, 2>, uint64_t>>;

std::string GetTestDataPath(const std::string& relative_path_to_file);
template <typename T>
gpuspatial::PinnedVector<T> ToVector(const rmm::cuda_stream_view& stream,
                                     const rmm::device_uvector<T>& d_vec) {
  gpuspatial::PinnedVector<T> vec(d_vec.size());

  thrust::copy(rmm::exec_policy_nosync(stream), d_vec.begin(), d_vec.end(), vec.begin());
  return vec;
}
template <typename T>
gpuspatial::PinnedVector<T> ToVector(const rmm::cuda_stream_view& stream,
                                     const gpuspatial::ArrayView<T>& arr) {
  gpuspatial::PinnedVector<T> vec(arr.size());

  thrust::copy(rmm::exec_policy_nosync(stream), arr.begin(), arr.end(), vec.begin());
  return vec;
}

}  // namespace TestUtils
#endif  // GPUSPATIAL_TEST_TEST_COMMON_HPP
