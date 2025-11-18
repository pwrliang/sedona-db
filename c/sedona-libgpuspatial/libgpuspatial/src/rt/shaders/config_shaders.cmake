
include(cmake/nvcuda_compile_module.cmake)

function(CONFIG_SHADERS SHADER_PTX_FILES)
    set(SHADER_POINT_TYPES "SHADER_POINT_FLOAT_2D;SHADER_POINT_DOUBLE_2D")

    set(SHADERS_DEPS "${PROJECT_SOURCE_DIR}/include/gpuspatial/geom"
            "${PROJECT_SOURCE_DIR}/include/gpuspatial/index/detail")

    set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/shaders_ptx")
    set(OPTIX_MODULE_EXTENSION ".ptx")
    set(OPTIX_PROGRAM_TARGET "--ptx")

    set(ALL_GENERATED_FILES "")

    foreach (POINT_TYPE IN LISTS SHADER_POINT_TYPES)
        NVCUDA_COMPILE_MODULE(
                SOURCES "${PROJECT_SOURCE_DIR}/src/rt/shaders/point_query.cu"
                DEPENDENCIES ${SHADERS_DEPS}
                TARGET_PATH "${OUTPUT_DIR}"
                PREFIX "${POINT_TYPE}_"
                EXTENSION "${OPTIX_MODULE_EXTENSION}"
                GENERATED_FILES PROGRAM_MODULES
                NVCC_OPTIONS "${OPTIX_PROGRAM_TARGET}"
                "--gpu-architecture=compute_75"
                "--relocatable-device-code=true"
                "--expt-relaxed-constexpr"
                "-Wno-deprecated-gpu-targets"
		        "-std=c++17"
                "-I${optix_SOURCE_DIR}/include"
                "-I${PROJECT_SOURCE_DIR}/include"
                "-D${POINT_TYPE}"
        )
        list(APPEND ALL_GENERATED_FILES ${PROGRAM_MODULES})

        NVCUDA_COMPILE_MODULE(
                SOURCES "${PROJECT_SOURCE_DIR}/src/rt/shaders/box_query_forward.cu"
                DEPENDENCIES ${SHADERS_DEPS}
                TARGET_PATH "${OUTPUT_DIR}"
                PREFIX "${POINT_TYPE}_"
                EXTENSION "${OPTIX_MODULE_EXTENSION}"
                GENERATED_FILES PROGRAM_MODULES
                NVCC_OPTIONS "${OPTIX_PROGRAM_TARGET}"
                "--gpu-architecture=compute_75"
                "--relocatable-device-code=true"
                "--expt-relaxed-constexpr"
                "-Wno-deprecated-gpu-targets"
		        "-std=c++17"
                "-I${optix_SOURCE_DIR}/include"
                "-I${PROJECT_SOURCE_DIR}/include"
                "-D${POINT_TYPE}"
        )
        list(APPEND ALL_GENERATED_FILES ${PROGRAM_MODULES})

        NVCUDA_COMPILE_MODULE(
                SOURCES "${PROJECT_SOURCE_DIR}/src/rt/shaders/box_query_backward.cu"
                DEPENDENCIES ${SHADERS_DEPS}
                TARGET_PATH "${OUTPUT_DIR}"
                PREFIX "${POINT_TYPE}_"
                EXTENSION "${OPTIX_MODULE_EXTENSION}"
                GENERATED_FILES PROGRAM_MODULES
                NVCC_OPTIONS "${OPTIX_PROGRAM_TARGET}"
                "--gpu-architecture=compute_75"
                "--relocatable-device-code=true"
                "--expt-relaxed-constexpr"
                "-Wno-deprecated-gpu-targets"
		        "-std=c++17"
                "-I${optix_SOURCE_DIR}/include"
                "-I${PROJECT_SOURCE_DIR}/include"
                "-D${POINT_TYPE}"
        )
        list(APPEND ALL_GENERATED_FILES ${PROGRAM_MODULES})

        NVCUDA_COMPILE_MODULE(
                SOURCES "${PROJECT_SOURCE_DIR}/src/rt/shaders/polygon_point_query.cu"
                DEPENDENCIES ${SHADERS_DEPS}
                TARGET_PATH "${OUTPUT_DIR}"
                PREFIX "${POINT_TYPE}_"
                EXTENSION "${OPTIX_MODULE_EXTENSION}"
                GENERATED_FILES PROGRAM_MODULES
                NVCC_OPTIONS "${OPTIX_PROGRAM_TARGET}"
                "--gpu-architecture=compute_75"
                "--relocatable-device-code=true"
                "--expt-relaxed-constexpr"
                "-Wno-deprecated-gpu-targets"
                "-std=c++17"
                "-I${optix_SOURCE_DIR}/include"
                "-I${PROJECT_SOURCE_DIR}/include"
                "-D${POINT_TYPE}"
        )
        list(APPEND ALL_GENERATED_FILES ${PROGRAM_MODULES})

        NVCUDA_COMPILE_MODULE(
                SOURCES "${PROJECT_SOURCE_DIR}/src/rt/shaders/multipolygon_point_query.cu"
                DEPENDENCIES ${SHADERS_DEPS}
                TARGET_PATH "${OUTPUT_DIR}"
                PREFIX "${POINT_TYPE}_"
                EXTENSION "${OPTIX_MODULE_EXTENSION}"
                GENERATED_FILES PROGRAM_MODULES
                NVCC_OPTIONS "${OPTIX_PROGRAM_TARGET}"
                "--gpu-architecture=compute_75"
                "--relocatable-device-code=true"
                "--expt-relaxed-constexpr"
                "-Wno-deprecated-gpu-targets"
                "-std=c++17"
                "-I${optix_SOURCE_DIR}/include"
                "-I${PROJECT_SOURCE_DIR}/include"
                "-D${POINT_TYPE}"
        )
        list(APPEND ALL_GENERATED_FILES ${PROGRAM_MODULES})
    endforeach ()
    set(${SHADER_PTX_FILES} ${ALL_GENERATED_FILES} PARENT_SCOPE)
endfunction()
