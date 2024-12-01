#include <optix.h>
#include <sutil/vec_math.h>
#include <cuda/GeometryData.h>
#include <cuda/util.h>
#include <vector_types.h>

static __forceinline__ __device__ void  PackPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
    // CUdeviceptr's size if 64 bit
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void* UnpackPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ float* GetTPayload()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<float*>(UnpackPointer(u0, u1));
}

static __forceinline__ __device__ float4* GetColorPayload()
{
    const uint32_t u0 = optixGetPayload_2();
    const uint32_t u1 = optixGetPayload_3();
    return reinterpret_cast<float4*>(UnpackPointer(u0, u1));
}

static __forceinline__ __device__ uint32_t* GetTailPayload()
{
    const uint32_t u0 = optixGetPayload_4();
    const uint32_t u1 = optixGetPayload_5();
    return reinterpret_cast<uint32_t*>(UnpackPointer(u0, u1));
}