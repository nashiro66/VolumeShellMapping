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

#include <optix.h>
#include <sutil/vec_math.h>
#include <cuda/BufferView.h>
#include <cuda/Light.h>
#include <cuda/whitted.h>
#include <sutil/Scene.h>

const unsigned int NUM_PAYLOAD_VALUES = 8u;
const unsigned int NUM_ATTRIBUTE_VALUES = 4u;
const unsigned int BUFFER_SIZE = 256u;
const unsigned int FRAMENUM = 101u;
const unsigned int SPECTRUMSIZE = 257u;

enum ObjectType
{
    ANYHIT_OBJECT = 1,
    CLOSEST_OBJECT = 2,
    ANY_OBJECT = 0xFF
};

enum MissRayType
{
    MISS_RAY_TYPE_ANYHIT = 0,
    MISS_RAY_TYPE_CLOSEST = 1,
    MISS_RAY_TYPE_COUNT = 2
};


enum HitRayType
{
    HIT_RAY_TYPE_ANY = 0,
    HIT_RAY_TYPE_COUNT = 1
};


struct LaunchParams
{
    unsigned int             width;
    unsigned int             height;
    unsigned int             subframe_index;

    float4*                  accum_buffer;
    float4*                  spectrum_map;
    uchar4*                  frame_buffer;

    float3                   eye;
    float3                   U;
    float3                   V;
    float3                   W;

    BufferView<Light>        lights;
    float3                   miss_color;
    OptixTraversableHandle   handle;

    unsigned int frameIndex;
};

struct VolumeMaterialData
{
    struct Lambert
    {
        float3 base_color;
    };

    struct Volume
    {
        float  opacity;
    };

    Volume  volume;
};

struct VolumeGeometryData
{
    struct Volume
    {
        void* grid;
    };

    Volume volume;
};

struct HitGroupData
{
    VolumeGeometryData volume_data[FRAMENUM];
    VolumeMaterialData volume_material_data;
    GeometryData geometry_data;
    MaterialData geometry_material_data;
};