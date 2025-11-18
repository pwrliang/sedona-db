# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
cmake_minimum_required(VERSION 3.30.4)

set(GPUSPATIAL_VERSION "0.1.0-SNAPSHOT")
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" GPUSPATIAL_BASE_VERSION
             "${GPUSPATIAL_VERSION}")

project(gpuspatial
        LANGUAGES CUDA C CXX
        VERSION "${GPUSPATIAL_BASE_VERSION}")

message(STATUS "Building gpuspatial version: ${PROJECT_VERSION}")
message(STATUS "Building using CMake version: ${CMAKE_VERSION}")

# =============================================================================
# Project-wide Settings
# =============================================================================
set(CMAKE_CXX_STANDARD 17)

option(GPUSPATIAL_BUILD_TESTS "Build tests" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

include(cmake/rapids_config.cmake)
include("${rapids-cmake-dir}/export/find_package_root.cmake")
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

rapids_cmake_build_type(Release)

# =============================================================================
# Fetch Dependencies
# =============================================================================
include(FetchContent)

rapids_cpm_init()
if(GPUSPATIAL_BUILD_TESTS)
  set(NANOARROW_IPC ON)
  set(NANOARROW_IPC_WITH_ZSTD ON)

  include(cmake/thirdparty/get_gtest.cmake)
  include(cmake/thirdparty/get_zstd.cmake)
endif()
include(cmake/thirdparty/get_nanoarrow.cmake)
include(cmake/thirdparty/get_geoarrow.cmake)
include(cmake/thirdparty/get_rmm.cmake)

if(NOT BUILD_SHARED_LIBS)
  include("${rapids-cmake-dir}/export/find_package_file.cmake")
  list(APPEND METADATA_KINDS BUILD INSTALL)
  list(APPEND
       dependencies
       nanoarrow
       zstd
       geoarrow)

  foreach(METADATA_KIND IN LISTS METADATA_KINDS)
    foreach(dep IN LISTS dependencies)
      rapids_export_package(${METADATA_KIND} ${dep} gpuspatial-exports)
    endforeach()
  endforeach()
endif()

# rapids dependencies
include(${rapids-cmake-dir}/cpm/rapids_logger.cmake)
rapids_cpm_rapids_logger(BUILD_EXPORT_SET gpuspatial-exports INSTALL_EXPORT_SET
                         gpuspatial-exports)
create_logger_macros(GPUSPATIAL "gpuspatial::default_logger()" include/gpuspatial)

fetchcontent_declare(OptiX
                     URL https://github.com/NVIDIA/optix-dev/archive/refs/tags/v8.0.0.zip
                     URL_HASH SHA256=c4b0ac2d2800ed35b4a2518f8db5ea40b279d6507db64e15c06c921d23d366a8
                     DOWNLOAD_EXTRACT_TIMESTAMP false)
fetchcontent_makeavailable(OptiX)

# Create an INTERFACE target for OptiX to manage its include directory
add_library(OptiX INTERFACE)
target_include_directories(OptiX
                           INTERFACE $<BUILD_INTERFACE:${optix_SOURCE_DIR}/include>
                                     $<INSTALL_INTERFACE:include> # Corresponds to <prefix>/include
)

# Set logging level
set(LIBGPUSPATIAL_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level.")
set_property(CACHE LIBGPUSPATIAL_LOGGING_LEVEL
             PROPERTY STRINGS
                      "TRACE"
                      "DEBUG"
                      "INFO"
                      "WARN"
                      "ERROR"
                      "CRITICAL"
                      "OFF")
message(VERBOSE
        "GPUSPATIAL: LIBGPUSPATIAL_LOGGING_LEVEL = '${LIBGPUSPATIAL_LOGGING_LEVEL}'.")

# =============================================================================
# Target Definition (gpuspatial)
# =============================================================================

include(src/rt/shaders/config_shaders.cmake)
config_shaders(PTX_FILES)

message("-- Config shader PTX files ${PTX_FILES}")

add_library(gpuspatial src/rt/rt_engine.cpp src/relate_engine.cu src/spatial_joiner.cu
                       ${PTX_FILES})

# Link libraries
target_link_libraries(gpuspatial
                      PUBLIC nanoarrow::nanoarrow
                             geoarrow
                             cuda
                             rmm::rmm
                             rapids_logger::rapids_logger
                             OptiX
                      PRIVATE zstd)

# Set include directories
target_include_directories(gpuspatial
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include> # For generated logger header
                                  $<INSTALL_INTERFACE:include> # Path for installed headers
                           PRIVATE src)

# Set compile options
target_compile_options(gpuspatial
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda
                               --expt-relaxed-constexpr>
                               -DGPUSPATIAL_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${LIBGPUSPATIAL_LOGGING_LEVEL}
)

add_library(gpuspatial_c src/gpuspatial_c.cc)
target_link_libraries(gpuspatial_c PUBLIC gpuspatial)

# =============================================================================
# Installation
# =============================================================================
include(GNUInstallDirs)

# Install OptiX headers
install(DIRECTORY ${optix_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install the .ptx shader files
set(GPUSPATIAL_SHADER_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/gpuspatial/shaders")
install(FILES ${PTX_FILES} DESTINATION ${GPUSPATIAL_SHADER_INSTALL_DIR})

# Install the library and public headers
install(TARGETS gpuspatial gpuspatial_c OptiX
        EXPORT gpuspatial-exports
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install public headers from the 'include' directory
# This assumes your public headers are in a subdirectory like 'include/gpuspatial/'
# and will install them to '<prefix>/include/gpuspatial/'
install(DIRECTORY include/gpuspatial/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gpuspatial)

rapids_export(INSTALL
              gpuspatial
              EXPORT_SET
              gpuspatial-exports
              GLOBAL_TARGETS
              gpuspatial
              VERSION
              ${PROJECT_VERSION}
              NAMESPACE
              gpuspatial::)

rapids_export(BUILD
              gpuspatial
              EXPORT_SET
              gpuspatial-exports
              GLOBAL_TARGETS
              gpuspatial
              VERSION
              ${PROJECT_VERSION}
              NAMESPACE
              gpuspatial::)

if(GPUSPATIAL_BUILD_TESTS)
  fetchcontent_declare(geos
                       URL https://github.com/libgeos/geos/archive/refs/tags/3.13.1.zip
                       URL_HASH SHA256=485568431802ab03c60e3359d3869fe008b025454d2cc6a16c3aed683bb56d05
                       DOWNLOAD_EXTRACT_TIMESTAMP false)

  fetchcontent_makeavailable(geos)

  add_library(geoarrow_geos test/geoarrow_geos/geoarrow_geos.c)
  target_link_libraries(geoarrow_geos
                        PUBLIC GEOS::geos_c
                        PRIVATE geoarrow)
endif()

if(GPUSPATIAL_BUILD_TESTS)
  enable_testing()

  add_executable(gpuspatial_testing_test test/gpuspatial_testing_test.cc)
  target_link_libraries(gpuspatial_testing_test
                        geoarrow
                        GTest::gtest_main
                        GTest::gmock_main
                        gpuspatial)

  add_executable(array_stream_test test/array_stream_test.cc test/array_stream.cc)
  target_link_libraries(array_stream_test
                        GTest::gtest_main
                        GTest::gmock_main
                        geoarrow
                        nanoarrow::nanoarrow
                        nanoarrow::nanoarrow_ipc)

  add_executable(loader_test test/array_stream.cc test/main.cc test/loader_test.cu)
  target_link_libraries(loader_test
                        cuda
                        GTest::gtest_main
                        GTest::gmock_main
                        gpuspatial
                        GEOS::geos
                        GEOS::geos_c
                        nanoarrow::nanoarrow_ipc)
  target_include_directories(loader_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_options(loader_test
                         PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda
                                 --expt-relaxed-constexpr>)

  add_executable(joiner_test test/array_stream.cc test/main.cc test/joiner_test.cu)
  target_link_libraries(joiner_test
                        cuda
                        GTest::gtest_main
                        GTest::gmock_main
                        gpuspatial
                        GEOS::geos
                        GEOS::geos_c
                        geoarrow_geos
                        nanoarrow::nanoarrow_ipc)
  target_include_directories(joiner_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_options(joiner_test
                         PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda
                                 --expt-relaxed-constexpr>)

  add_executable(relate_test test/main.cc test/array_stream.cc test/related_test.cu)
  target_link_libraries(relate_test
                        PRIVATE cuda
                                GTest::gtest_main
                                GTest::gmock_main
                                gpuspatial
                                GEOS::geos
                                nanoarrow::nanoarrow
                                nanoarrow::nanoarrow_ipc)
  target_compile_options(relate_test
                         PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda
                                 --expt-relaxed-constexpr>)

  add_executable(c_wrapper_test test/main.cc test/c_wrapper_test.cc test/array_stream.cc)
  target_link_libraries(c_wrapper_test PRIVATE GTest::gtest_main GTest::gmock_main
                                               gpuspatial_c nanoarrow::nanoarrow_ipc)

  include(GoogleTest)

  gtest_discover_tests(gpuspatial_testing_test)
  gtest_discover_tests(array_stream_test)
  gtest_discover_tests(loader_test)
  gtest_discover_tests(joiner_test)
  gtest_discover_tests(relate_test)
endif()
