cmake_minimum_required(VERSION 3.30.4)

set(GPUSPATIAL_VERSION "0.1.0-SNAPSHOT")
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" GPUSPATIAL_BASE_VERSION "${GPUSPATIAL_VERSION}")

project(gpuspatial LANGUAGES CUDA C CXX VERSION "${GPUSPATIAL_BASE_VERSION}")

message(STATUS "Building gpuspatial version: ${PROJECT_VERSION}")
message(STATUS "Building using CMake version: ${CMAKE_VERSION}")

# =============================================================================
# Project-wide Settings
# =============================================================================
set(CMAKE_CXX_STANDARD 17)

option(GPUSPATIAL_BUILD_TESTS "Build tests" OFF)
option(GPUSPATIAL_BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

include(cmake/rapids_config.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

rapids_cmake_build_type(Release)

# =============================================================================
# Fetch Dependencies
# =============================================================================
include(FetchContent)

set(NANOARROW_IPC ON)
set(NANOARROW_IPC_WITH_ZSTD ON) # Make sure you have libzstd-dev
# BUILD_TESTS is OFF for all dependencies to speed up build
FetchContent_Declare(
        nanoarrow
        URL https://codeload.github.com/apache/arrow-nanoarrow/zip/4bfd77c82793eb91c804d6f893f7ab324e0105a0
        URL_HASH SHA256=7cd772be4438e9699efab06d32c1ec817319ccdbe81a3495bfd6c9a80554c103
        DOWNLOAD_EXTRACT_TIMESTAMP false
        OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(nanoarrow)
rapids_export_package(BUILD nanoarrow gpuspatial-exports)
rapids_export_package(INSTALL nanoarrow gpuspatial-exports)

FetchContent_Declare(
        geoarrow
        URL https://github.com/geoarrow/geoarrow-c/archive/4e86dea238cf251fc235b98d2fa6976cc6446dd0.zip
        URL_HASH SHA256=a642873190712bf8301378400ff36866b188c6ed7b6565a1f1374f74caf99768
        DOWNLOAD_EXTRACT_TIMESTAMP false
        OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(geoarrow)
rapids_export_package(BUILD geoarrow gpuspatial-exports)
rapids_export_package(INSTALL geoarrow gpuspatial-exports)

FetchContent_Declare(
        OptiX
        URL https://github.com/NVIDIA/optix-dev/archive/refs/tags/v8.0.0.zip
        URL_HASH SHA256=c4b0ac2d2800ed35b4a2518f8db5ea40b279d6507db64e15c06c921d23d366a8
        DOWNLOAD_EXTRACT_TIMESTAMP false
)
FetchContent_MakeAvailable(OptiX)

# Create an INTERFACE target for OptiX to manage its include directory
add_library(OptiX INTERFACE)
target_include_directories(OptiX INTERFACE
        $<BUILD_INTERFACE:${optix_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include> # Corresponds to <prefix>/include
)

rapids_find_package(
        CUDAToolkit REQUIRED
        BUILD_EXPORT_SET gpuspatial-exports
        INSTALL_EXPORT_SET gpuspatial-exports)

# rapids dependencies
rapids_cpm_init()
include(${rapids-cmake-dir}/cpm/rapids_logger.cmake)
rapids_cpm_rapids_logger(BUILD_EXPORT_SET gpuspatial-exports INSTALL_EXPORT_SET gpuspatial-exports)
create_logger_macros(GPUSPATIAL "gpuspatial::default_logger()" include/gpuspatial)

include(cmake/get_rmm.cmake)

# Set logging level
set(LIBGPUSPATIAL_LOGGING_LEVEL
        "INFO"
        CACHE STRING "Choose the logging level."
)
set_property(
        CACHE LIBGPUSPATIAL_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL"
        "OFF"
)
message(VERBOSE "GPUSPATIAL: LIBGPUSPATIAL_LOGGING_LEVEL = '${LIBGPUSPATIAL_LOGGING_LEVEL}'.")

# =============================================================================
# Target Definition (gpuspatial)
# =============================================================================

include(src/rt/shaders/config_shaders.cmake)
CONFIG_SHADERS(PTX_FILES)

message("-- Config shader PTX files ${PTX_FILES}")

add_library(gpuspatial src/rt/rt_engine.cpp src/relate_engine.cu src/spatial_joiner.cu ${PTX_FILES})

# Link libraries
target_link_libraries(gpuspatial PUBLIC nanoarrow::nanoarrow geoarrow cuda rmm::rmm rapids_logger::rapids_logger OptiX)

# Set include directories
target_include_directories(gpuspatial
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include> # For generated logger header
        $<INSTALL_INTERFACE:include> # Path for installed headers
        PRIVATE
        src
)

# Set compile options
target_compile_options(gpuspatial PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr> -DGPUSPATIAL_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${LIBGPUSPATIAL_LOGGING_LEVEL})

add_library(gpuspatial_c src/gpuspatial_c.cc)
target_link_libraries(gpuspatial_c PUBLIC gpuspatial)

# =============================================================================
# Installation
# =============================================================================
include(GNUInstallDirs)

# Install OptiX headers
install(DIRECTORY ${optix_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install the .ptx shader files
set(GPUSPATIAL_SHADER_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/gpuspatial/shaders")
install(FILES ${PTX_FILES}
        DESTINATION ${GPUSPATIAL_SHADER_INSTALL_DIR}
)

# Install the library and public headers
install(TARGETS gpuspatial gpuspatial_c OptiX
        EXPORT gpuspatial-exports
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install public headers from the 'include' directory
# This assumes your public headers are in a subdirectory like 'include/gpuspatial/'
# and will install them to '<prefix>/include/gpuspatial/'
install(DIRECTORY include/gpuspatial/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gpuspatial
)

rapids_export(
        INSTALL gpuspatial
        EXPORT_SET gpuspatial-exports
        GLOBAL_TARGETS gpuspatial
        VERSION ${PROJECT_VERSION}
        NAMESPACE gpuspatial::)

rapids_export(
        BUILD gpuspatial
        EXPORT_SET gpuspatial-exports
        GLOBAL_TARGETS gpuspatial
        VERSION ${PROJECT_VERSION}
        NAMESPACE gpuspatial::)

if (GPUSPATIAL_BUILD_TESTS OR GPUSPATIAL_BUILD_BENCHMARK)
    FetchContent_Declare(
            geos
            URL https://github.com/libgeos/geos/archive/refs/tags/3.13.1.zip
            URL_HASH
            SHA256=485568431802ab03c60e3359d3869fe008b025454d2cc6a16c3aed683bb56d05
            DOWNLOAD_EXTRACT_TIMESTAMP false
    )

    FetchContent_MakeAvailable(geos)

    include_directories(thirdparty)
    add_library(geoarrow_geos thirdparty/geoarrow_geos/geoarrow_geos.c)
    target_link_libraries(
            geoarrow_geos
            PUBLIC GEOS::geos_c
            PRIVATE geoarrow)
endif ()

if (GPUSPATIAL_BUILD_TESTS)
    find_package(GTest CONFIG REQUIRED)

    enable_testing()

    add_executable(gpuspatial_testing_test
            test/gpuspatial_testing_test.cc)
    target_link_libraries(gpuspatial_testing_test geoarrow GTest::gtest_main GTest::gmock_main
            gpuspatial)

    add_executable(array_stream_test test/array_stream_test.cc test/array_stream.cc)
    target_link_libraries(array_stream_test GTest::gtest_main GTest::gmock_main geoarrow nanoarrow::nanoarrow nanoarrow::nanoarrow_ipc)

    add_executable(loader_test test/array_stream.cc test/main.cc test/loader_test.cu)
    target_link_libraries(loader_test cuda GTest::gtest_main GTest::gmock_main gpuspatial GEOS::geos GEOS::geos_c nanoarrow::nanoarrow_ipc)
    target_include_directories(loader_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_options(loader_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)

    add_executable(joiner_test test/array_stream.cc test/main.cc test/joiner_test.cu)
    target_link_libraries(joiner_test cuda GTest::gtest_main GTest::gmock_main gpuspatial GEOS::geos GEOS::geos_c geoarrow_geos nanoarrow::nanoarrow_ipc)
    target_include_directories(joiner_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_options(joiner_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)

    add_executable(index_test test/main.cc test/index_test.cu)
    target_link_libraries(index_test cuda GTest::gtest_main GTest::gmock_main gpuspatial)
    target_include_directories(index_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_options(index_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)

    add_executable(relate_test test/main.cc test/array_stream.cc test/related_test.cu)
    target_link_libraries(relate_test PRIVATE cuda GTest::gtest_main GTest::gmock_main gpuspatial GEOS::geos nanoarrow::nanoarrow nanoarrow::nanoarrow_ipc)
    target_compile_options(relate_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)


    add_executable(c_wrapper_test test/main.cc test/c_wrapper_test.cc test/array_stream.cc)
    target_link_libraries(c_wrapper_test PRIVATE GTest::gtest_main GTest::gmock_main gpuspatial_c nanoarrow::nanoarrow_ipc)

    include(GoogleTest)

    gtest_discover_tests(gpuspatial_testing_test)
    gtest_discover_tests(array_stream_test)
    gtest_discover_tests(loader_test)
    gtest_discover_tests(joiner_test)
    gtest_discover_tests(index_test)
    gtest_discover_tests(relate_test)
endif ()


if (GPUSPATIAL_BUILD_BENCHMARK)
    find_package(gflags CONFIG REQUIRED)
    find_package(nlohmann_json CONFIG REQUIRED)

    find_package(Arrow REQUIRED)
    find_package(Parquet REQUIRED)
    set(BOOST_INCLUDE_LIBRARIES math geometry)
    FetchContent_Declare(
            Boost
            URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.xz
            URL_MD5 893b5203b862eb9bbd08553e24ff146a
            DOWNLOAD_EXTRACT_TIMESTAMP ON
    )
    FetchContent_MakeAvailable(Boost)

    add_executable(benchmark benchmark/benchmark.cpp
            benchmark/run_queries.cu
            benchmark/flags.cpp
            #            benchmark/baseline/boost_index.cpp
            benchmark/baseline/geos_index.cpp
            #            benchmark/baseline/wkb.cpp
    )
    target_link_libraries(benchmark PRIVATE cuda geoarrow gpuspatial GEOS::geos GEOS::geos_c geoarrow_geos gflags
            Arrow::arrow_static Parquet::parquet_static nlohmann_json::nlohmann_json Boost::geometry)
    target_include_directories(benchmark PRIVATE ${CUDAToolkit_INCLUDE_DIRS} thirdparty benchmark)
    target_compile_options(benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
endif ()
