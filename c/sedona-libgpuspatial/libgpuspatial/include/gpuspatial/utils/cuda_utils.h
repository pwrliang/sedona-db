#pragma once
#define MAX_BLOCK_SIZE (256)
#define WARP_SIZE (32)
#if defined(__CUDACC__) || defined(__CUDABE__)
#define DEV_HOST __device__ __host__
#define DEV_HOST_INLINE __device__ __host__ __forceinline__
#define DEV_INLINE __device__ __forceinline__
#define CONST_STATIC_INIT(...)

#define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)
#define TOTAL_THREADS_1D (gridDim.x * blockDim.x)

#else
#define DEV_HOST
#define DEV_HOST_INLINE
#define DEV_INLINE
#define CONST_STATIC_INIT(...) = __VA_ARGS__

#define THRUST_TO_CUPTR(x) (reinterpret_cast<CUdeviceptr>(thrust::raw_pointer_cast(x)))
#endif


