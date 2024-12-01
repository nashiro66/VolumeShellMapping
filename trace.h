//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include "volume.h"

#include <cuda_runtime.h>
#include "nanovdb/NanoVDB.h"


static __forceinline__ __device__ void TraceAnyhit(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    unsigned int           mask,
    uint32_t* t0, uint32_t* t1,
    uint32_t* colorAddress0, uint32_t* colorAddress1,
    uint32_t* tail0, uint32_t* tail1)
{
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,
        mask,
        OPTIX_RAY_FLAG_NONE,
        HIT_RAY_TYPE_ANY,
        HIT_RAY_TYPE_COUNT,
        MISS_RAY_TYPE_ANYHIT,
        *t0, *t1, *colorAddress0, *colorAddress1, *tail0, *tail1);
}

static __forceinline__ __device__ void TraceClosest(
    OptixTraversableHandle handle,
    float3                 ray_origin,
    float3                 ray_direction,
    float                  tmin,
    float                  tmax,
    unsigned int           mask,
    uint32_t* t0, uint32_t* t1)
{
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,  // rayTime
        mask,
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        HIT_RAY_TYPE_ANY,
        HIT_RAY_TYPE_COUNT,
        MISS_RAY_TYPE_CLOSEST,
        *t0, *t1
    );
}
