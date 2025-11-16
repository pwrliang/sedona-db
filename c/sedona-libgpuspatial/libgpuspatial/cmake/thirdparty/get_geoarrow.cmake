# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds geoarrow and sets any additional necessary environment variables.
function(find_and_configure_geoarrow)
    if(NOT BUILD_SHARED_LIBS)
        set(_exclude_from_all EXCLUDE_FROM_ALL FALSE)
    else()
        set(_exclude_from_all EXCLUDE_FROM_ALL TRUE)
    endif()

    # Currently we need to always build geoarrow so we don't pickup a previous installed version
    set(CPM_DOWNLOAD_geoarrow ON)
    rapids_cpm_find(
            geoarrow geoarrow-c-python-0.3.1
            GLOBAL_TARGETS geoarrow
            CPM_ARGS
            GIT_REPOSITORY https://github.com/geoarrow/geoarrow-c.git
            GIT_TAG eae46da505d9a5a8c156fc6bbb80798f2cb4a3d0
            GIT_SHALLOW FALSE
            OPTIONS "BUILD_SHARED_LIBS OFF" ${_exclude_from_all}
    )
    set_target_properties(geoarrow PROPERTIES POSITION_INDEPENDENT_CODE ON)
    rapids_export_find_package_root(BUILD geoarrow "${geoarrow_BINARY_DIR}" EXPORT_SET gpuspatial-exports)
endfunction()

find_and_configure_geoarrow()