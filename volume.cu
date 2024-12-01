#include <optix.h>
#include <ratio>
#include <sutil/vec_math.h>
#include <vector_types.h>
#include <cuda/random.h>
#include <cuda/helpers.h>
#include <cuda/LocalGeometry.h>
#include <cuda/LocalShading.h>

#include "hdda.h"
#include "trace.h"
#include "payload.h"
#include "prismIntersection.h"

__constant__ LaunchParams params;
__constant__ float eps = 1e-7f;
__constant__ float eeps = 1e-3f;
__constant__ float minDepth = 0.01f;
__constant__ float maxDepth = 1e16f;

// ----------------------------------------------------------------------------
// HDDA Func
// ----------------------------------------------------------------------------
static __device__ float SolveQuadraticEquation(float a, float b, float c, float hCur, float hDir, float hIn, float hOut)
{
    // TODO:
    // can be simplified
    float discriminant = b * b - 4 * a * c;
    if (discriminant >= 0)
    {
        const float delta = sqrt(discriminant);
        int sign = -1;
        if (b > 0)
        {
            sign = 1;
        }
        const float q = -0.5f * (b + sign * delta);
        const float x2 = q / a;
        const float x1 = c / q;
        float aa = fmin(x1, x2);
        float bb = fmax(x1, x2);

        if (((hIn <= aa && aa <= hOut) || (hOut <= aa && aa <= hIn)) &&
            ((hIn <= bb && bb <= hOut) || (hOut <= bb && bb <= hIn)))
        {
            if (hDir > 0.0f)
            {
                if (hCur < aa && hCur < bb)
                {
                    if (aa < bb)
                    {
                        return aa;
                    }
                    else
                    {
                        return bb;
                    }
                }
                else if (hCur < aa)
                {
                    return aa;
                }
                else if (hCur < bb)
                {
                    return bb;
                }
            }
            else
            {
                if (aa < hCur && bb < hCur)
                {
                    if (aa > bb)
                    {
                        return aa;
                    }
                    else
                    {
                        return bb;
                    }
                }
                else if (hCur > aa)
                {
                    return aa;
                }
                else if (hCur > bb)
                {
                    return bb;
                }
            }
        }
        else if ((hIn <= aa && aa <= hOut) || (hOut <= aa && aa <= hIn))
        {
            return aa;
        }
        else if ((hIn <= bb && bb <= hOut) || (hOut <= bb && bb <= hIn))
        {
            return bb;
        }
    }
    return hOut;
}

static __device__ float SolveConicQuadraticEquation(
    float alpha2, float beta2, float alphabeta, float alpha1, float beta1, float c0,
    float alpha2inbeta2, float alphainbeta, float UV0, float UV01, float UV02,
    float wantUV0, float wantUV01, float wantUV02, float uv, float& alpha, float& beta,
    float tCurrent, float3 ray_dir, float3 ray_orig, float3 s0, float3 s1, float3 s2)
{
    // TODO:
    // can be simplified
    const float alphainbeta2 = -2 * UV01 * (uv - UV0) / (UV02 * UV02);
    const float cinbeta2 = (uv - UV0) / UV02 * (uv - UV0) / UV02;
    const float cinbeta = (uv - UV0) / UV02;
    float a = alpha2 + beta2 * alpha2inbeta2 + alphabeta * alphainbeta;
    float b = beta2 * alphainbeta2 + alphabeta * cinbeta + alpha1 + beta1 * alphainbeta;
    float c = beta2 * cinbeta2 + beta1 * cinbeta + c0;
    float start = 0.0f;
    float end = 1.0f;
    float discriminant = b * b - 4 * a * c;
    if (discriminant >= 0)
    {
        const float delta = sqrt(discriminant);
        int sign = -1;
        if (b > 0)
        {
            sign = 1;
        }
        const float q = -0.5f * (b + sign * delta);
        const float x2 = q / a;
        const float x1 = c / q;
        float aa = fmin(x1, x2);
        float bb = fmax(x1, x2);

        if (((start <= aa && aa <= end) || (end <= aa && aa <= start)) &&
            ((start <= bb && bb <= end) || (end <= bb && bb <= start)))
        {
            float alphaCandidate1 = aa;
            float betaCandidate1 = (-UV01 * alphaCandidate1 + uv - UV0) / UV02;
            float tCandidate1 = dot(ray_dir, (1 - alphaCandidate1 - betaCandidate1) * s0 + alphaCandidate1 * s1 + betaCandidate1 * s2 - ray_orig);
            float alphaCandidate2 = bb;
            float betaCandidate2 = (-UV01 * alphaCandidate2 + uv - UV0) / UV02;
            float tCandidate2 = dot(ray_dir, (1 - alphaCandidate2 - betaCandidate2) * s0 + alphaCandidate2 * s1 + betaCandidate2 * s2 - ray_orig);
            if (tCurrent < tCandidate1 && tCurrent < tCandidate2)
            {
                if (tCandidate1 < tCandidate2)
                {
                    alpha = aa;
                }
                else
                {
                    alpha = bb;
                }
            }
            else if (tCurrent < tCandidate1)
            {
                alpha = aa;
            }
            else if (tCurrent < tCandidate2)
            {
                alpha = bb;
            }
        }
        else if ((start <= aa && aa <= end) || (end <= aa && aa <= start))
        {
            alpha = aa;
        }
        else if ((start <= bb && bb <= end) || (end <= bb && bb <= start))
        {
            alpha = bb;
        }
        else
        {
            alpha = end;
        }
    }
    else
    {
        alpha = end;
    }
    beta = (-UV01 * alpha + uv - UV0) / UV02;
    return  wantUV01 * alpha + wantUV02 * beta + wantUV0;
}

template<typename AccT>
inline __device__ float4 transmittanceHDDA(
    float hin, float hout, const nanovdb::FloatGrid* flameGrid, AccT& flameAcc, const nanovdb::FloatGrid* tempGrid, AccT& tempAcc, float opacity,
    float3 ray_orig, float3 ray_dir, float tStart, float tEnd, float3 triangle[3], float3 normals[3], int prismIdx)
{
    // https://github.com/shinjiogaki/nonlinear-ray-tracing/blob/main/src/shell.h as the reference
    ray_dir = normalize(ray_dir);
    const float3 axes[3] = { make_float3(1,0,0), make_float3(0,1,0), make_float3(0,0,1) };
    const float3 ex = (ray_dir.x < ray_dir.y && ray_dir.x < ray_dir.z) ? axes[0] :
        (ray_dir.y < ray_dir.z) ? axes[1] : axes[2];
    const float3 e1 = normalize(cross(ray_dir, ex));
    const float3 e0 = normalize(cross(e1, ray_dir));

    float e0P01 = dot(e0, triangle[1] - triangle[0]);
    float e0P02 = dot(e0, triangle[2] - triangle[0]);
    float e1P01 = dot(e1, triangle[1] - triangle[0]);
    float e1P02 = dot(e1, triangle[2] - triangle[0]);

    float e0N01 = dot(e0, normals[1] - normals[0]);
    float e0N02 = dot(e0, normals[2] - normals[0]);
    float e1N01 = dot(e1, normals[1] - normals[0]);
    float e1N02 = dot(e1, normals[2] - normals[0]);

    float e0P0o = dot(e0, ray_orig - triangle[0]);
    float e0N0 = dot(e0, normals[0]);
    float e1P0o = dot(e1, ray_orig - triangle[0]);
    float e1N0 = dot(e1, normals[0]);

    // alpha(h)
    const auto a2 = e0N02 * e1N0 - e1N02 * e0N0;
    const auto a1 = e1N02 * e0P0o - e1P02 * e0N0 + e0P02 * e1N0 - e0N02 * e1P0o;
    const auto a0 = e1P02 * e0P0o - e0P02 * e1P0o;

    // beta(h)
    const auto b2 = e1N01 * e0N0 - e0N01 * e1N0;
    const auto b1 = e1P01 * e0N0 - e1N01 * e0P0o + e0N01 * e1P0o - e0P01 * e1N0;
    const auto b0 = e0P01 * e1P0o - e1P01 * e0P0o;

    // d(h)
    const auto d2 = e0N01 * e1N02 - e0N02 * e1N01;
    const auto d1 = e0P01 * e1N02 + e0N01 * e1P02 - e0P02 * e1N01 - e0N02 * e1P01;
    const auto d0 = e0P01 * e1P02 - e0P02 * e1P01;
    float2 UV0;
    float2 UV01;
    float2 UV02;

    // simple fire obtained by https://jangafx.com/software/embergen/download/free-vdb-animations
    // necessary to convert vdb data to nanovdb data by https://github.com/AcademySoftwareFoundation/openvdb
    float uLen = 54.5f;
    float vLen = 50.5f;
    float2 uvwCenter = make_float2(-2.0f, -24.0f);
    const float scale = 120.0f;
    const float wCenter = 128.0f;

    if (prismIdx == 0)
    {
        UV0 = make_float2(uLen, -vLen) + uvwCenter;
        UV01 = make_float2(-2 * uLen, 0.0f);
        UV02 = make_float2(-2 * uLen, 2 * vLen);
    }
    else
    {
        UV0 = make_float2(uLen, -vLen) + uvwCenter;
        UV01 = make_float2(0.0f, 2 * vLen);
        UV02 = make_float2(-2 * uLen, 2 * vLen);
    }

    const auto u2 = a2 * UV01.x + b2 * UV02.x + d2 * UV0.x;
    const auto u1 = a1 * UV01.x + b1 * UV02.x + d1 * UV0.x;
    const auto u0 = a0 * UV01.x + b0 * UV02.x + d0 * UV0.x;
    const auto v2 = a2 * UV01.y + b2 * UV02.y + d2 * UV0.y;
    const auto v1 = a1 * UV01.y + b1 * UV02.y + d1 * UV0.y;
    const auto v0 = a0 * UV01.y + b0 * UV02.y + d0 * UV0.y;

    // conic curve term
    const float alpha2 = -e1N01 * e0P01 + e0N01 * e1P01;
    const float beta2 = -e1N02 * e0P02 + e0N02 * e1P02;
    const float alphabeta = -e1N01 * e0P02 - e1N02 * e0P01 + e0N01 * e1P02 + e0N02 * e1P01;
    const float alpha1 = e1N01 * e0P0o - e1N0 * e0P01 - e0N01 * e1P0o + e0N0 * e1P01;
    const float beta1 = e1N02 * e0P0o - e1N0 * e0P02 - e0N02 * e1P0o + e0N0 * e1P02;
    const float c0 = e1N0 * e0P0o - e0N0 * e1P0o;
    const float2 alpha2inbeta2 = UV01 / UV02 * UV01 / UV02;
    const float2 alphainbeta = -UV01 / UV02;

    float4 rgbt = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

    // HDDA
    float h, denom, alpha, beta;
    nanovdb::Vec3f uvw, uvwCandidateU, uvwCandidateV, uvwCandidateW;
    float hU, hV, hW;
    float alphaU, alphaV, alphaW;
    float betaU, betaV, betaW;
    float3 s0, s1, s2;
    float hDir = 1.0f;

    // hDir is uniquely determined
    h = hin;
    if (hout < hin)
    {
        hDir = -1.0f;
    }

    // initial barycentric coordinates
    s0 = triangle[0] + normals[0] * h;
    s1 = triangle[1] + normals[1] * h;
    s2 = triangle[2] + normals[2] * h;
    auto I = ray_orig + ray_dir * tStart;
    auto v01 = s1 - s0;
    auto v02 = s2 - s0;
    auto v0I = I - s0;
    float d00 = dot(v01, v01);
    float d01 = dot(v01, v02);
    float d11 = dot(v02, v02);
    float d20 = dot(v0I, v01);
    float d21 = dot(v0I, v02);
    float deno = d00 * d11 - d01 * d01;
    alpha = (d11 * d20 - d01 * d21) / deno;
    beta = (d00 * d21 - d01 * d20) / deno;

    // initial uvw
    denom = d2 * h * h + d1 * h + d0;
    uvw = nanovdb::Vec3f(
        UV01.x * alpha + UV02.x * beta + UV0.x,
        UV01.y * alpha + UV02.y * beta + UV0.y,
        h * scale - wCenter);
    uvw = flameGrid->worldToIndex(uvw);

    // TODO:
    // The interval where duvw changes sign can be determined in advance
    // RMIP: Displacement ray tracing via inversion and oblong bounding
    // https://dl.acm.org/doi/10.1145/3610548.3618182
    nanovdb::Vec3f duvw(
        ((2.0f * u2 * h + u1) / denom - (u2 * h * h + u1 * h + u0) / denom * (2.0f * d2 * h + d1) / denom) * hDir,
        ((2.0f * v2 * h + v1) / denom - (v2 * h * h + v1 * h + v0) / denom * (2.0f * d2 * h + d1) / denom) * hDir,
        hDir);
    duvw = flameGrid->worldToIndexDir(duvw);

    nanovdb::Coord ijk = nanovdb::RoundDown<nanovdb::Coord>(uvw);
    int dim = flameAcc.getDim(ijk);

    HDDA hdda(uvw, duvw, dim);
    float density = flameAcc.getValue(ijk) * opacity;
    float tPrevious = dot(ray_dir, (1 - alpha - beta) * s0 + alpha * s1 + beta * s2 - ray_orig);
    while (tPrevious < tEnd - 0.1f && 0.0f < rgbt.w)
    {

        int axis = hdda.GetAxis();
        if (axis == HDDA::u || axis == HDDA::all)
        {
            // param: u
            uvwCandidateU[HDDA::u] = hdda.GetParamCandidate(HDDA::u);
            uvwCandidateU = flameGrid->indexToWorld(uvwCandidateU);
            float a = uvwCandidateU[HDDA::u] * d2 - u2;
            float b = uvwCandidateU[HDDA::u] * d1 - u1;
            float c = uvwCandidateU[HDDA::u] * d0 - u0;
            hU = SolveQuadraticEquation(a, b, c, h, hDir, hin, hout);
            uvwCandidateU[HDDA::w] = hU * scale - wCenter;

            s0 = triangle[0] + normals[0] * hU;
            s1 = triangle[1] + normals[1] * hU;
            s2 = triangle[2] + normals[2] * hU;
            uvwCandidateU[HDDA::v] =
                SolveConicQuadraticEquation(
                    alpha2, beta2, alphabeta, alpha1, beta1, c0,
                    alpha2inbeta2.x, alphainbeta.x,
                    UV0.x, UV01.x, UV02.x,
                    UV0.y, UV01.y, UV02.y, uvwCandidateU[HDDA::u],
                    alphaU, betaU,
                    tPrevious, ray_dir, ray_orig, s0, s1, s2);
            uvwCandidateU = flameGrid->worldToIndex(uvwCandidateU);
        }
        if (axis == HDDA::v || axis == HDDA::all)
        {
            // param: v
            uvwCandidateV[HDDA::v] = hdda.GetParamCandidate(HDDA::v);
            uvwCandidateV = flameGrid->indexToWorld(uvwCandidateV);
            float a = uvwCandidateV[HDDA::v] * d2 - v2;
            float b = uvwCandidateV[HDDA::v] * d1 - v1;
            float c = uvwCandidateV[HDDA::v] * d0 - v0;
            hV = SolveQuadraticEquation(a, b, c, h, hDir, hin, hout);
            uvwCandidateV[HDDA::w] = hV * scale - wCenter;

            s0 = triangle[0] + normals[0] * hV;
            s1 = triangle[1] + normals[1] * hV;
            s2 = triangle[2] + normals[2] * hV;
            uvwCandidateV[HDDA::u] =
                SolveConicQuadraticEquation(
                    alpha2, beta2, alphabeta, alpha1, beta1, c0,
                    alpha2inbeta2.y, alphainbeta.y,
                    UV0.y, UV01.y, UV02.y,
                    UV0.x, UV01.x, UV02.x, uvwCandidateV[HDDA::v],
                    alphaV, betaV,
                    tPrevious, ray_dir, ray_orig, s0, s1, s2);
            uvwCandidateV = flameGrid->worldToIndex(uvwCandidateV);
        }
        if (axis == HDDA::w || axis == HDDA::all)
        {
            // param: w
            uvwCandidateW[HDDA::w] = hdda.GetParamCandidate(HDDA::w);
            uvwCandidateW = flameGrid->indexToWorld(uvwCandidateW);
            hW = (uvwCandidateW[HDDA::w] + wCenter) / scale;

            denom = d2 * hW * hW + d1 * hW + d0;
            uvwCandidateW[HDDA::u] = (u2 * hW * hW + u1 * hW + u0) / denom;
            uvwCandidateW[HDDA::v] = (v2 * hW * hW + v1 * hW + v0) / denom;
            alphaW = (a2 * hW * hW + a1 * hW + a0) / denom;
            betaW = (b2 * hW * hW + b1 * hW + b0) / denom;
            uvwCandidateW = flameGrid->worldToIndex(uvwCandidateW);
        }
        hdda.Step(
            uvw, uvwCandidateU, uvwCandidateV, uvwCandidateW,
            h, hU, hV, hW,
            alpha, alphaU, alphaV, alphaW,
            beta, betaU, betaV, betaW);

        // The interval where duvw changes sign can be determined in advance
        Update(duvw, u2, u1, u0, v2, v1, v0, denom)

        ijk = nanovdb::RoundDown<nanovdb::Coord>(uvw);
        dim = tempAcc.getDim(ijk);

        s0 = triangle[0] + normals[0] * h;
        s1 = triangle[1] + normals[1] * h;
        s2 = triangle[2] + normals[2] * h;

        float tCurrent = dot(ray_dir, (1 - alpha - beta) * s0 + alpha * s1 + beta * s2 - ray_orig);
        float distance = abs(tCurrent - tPrevious);
        tPrevious = tCurrent;
        rgbt.w *= expf(-density * distance);

        density = flameAcc.getValue(ijk) * opacity;
        hdda.Update(uvw, duvw, dim);
    }
    return rgbt;
}

// ----------------------------------------------------------------------------
// Raygen program
// ----------------------------------------------------------------------------
extern "C" __global__ void __raygen__pinhole()
{
    const uint3  launch_idx = optixGetLaunchIndex();
    const uint3  launch_dims = optixGetLaunchDimensions();
    const float3 eye = params.eye;
    const float3 U = params.U;
    const float3 V = params.V;
    const float3 W = params.W;
    const int    subframe_index = params.subframe_index;

    unsigned int seed = tea<4>(launch_idx.y * launch_dims.x + launch_idx.x, subframe_index);
    const float2 subpixel_jitter = subframe_index == 0 ? make_float2(0.5f, 0.5f) : make_float2(rnd(seed), rnd(seed));

    const float2 d = 2.0f *
        make_float2((static_cast<float>(launch_idx.x) + subpixel_jitter.x) / static_cast<float>(launch_dims.x),
            (static_cast<float>(launch_idx.y) + subpixel_jitter.y) / static_cast<float>(launch_dims.y)) - 1.0f;
    const float3 ray_direction = normalize(d.x * U + d.y * V + W);
    const float3 ray_origin = eye;

    float3 result = make_float3(0.0);
    float curTrancemittance = 1.0f;

    float t[BUFFER_SIZE];
    float4 colors[BUFFER_SIZE];
    uint32_t tail = 0;
    for (uint32_t i = 0; i < BUFFER_SIZE; i++)
    {
        t[i] = 0.0f;
        colors[i] = make_float4(0);
    }
    uint32_t t0, t1;
    PackPointer(t, t0, t1);
    uint32_t colorAddress0, colorAddress1;
    PackPointer(colors, colorAddress0, colorAddress1);
    uint32_t tail0, tail1;
    PackPointer(&tail, tail0, tail1);

    // trace ray
    t[0] = maxDepth;
    TraceClosest(
        params.handle, ray_origin, ray_direction,
        minDepth,  // tmin
        maxDepth,  // tmax
        CLOSEST_OBJECT,
        &t0, &t1);

    TraceAnyhit(
        params.handle, ray_origin, ray_direction,
        minDepth,  // tmin
        t[0],      // tmax
        ANYHIT_OBJECT,
        &t0, &t1,
        &colorAddress0, &colorAddress1,
        &tail0, &tail1);

    // sort by proximity
    for (int i = 0; i < tail; i++) {
        for (int j = tail - 1; i <= j; j--)
        {
            if (t[j] < t[j - 1]) {
                const float tempT = t[j - 1];
                const float4 tmpColor = colors[j - 1];
                t[j - 1] = t[j];
                t[j] = tempT;
                colors[j - 1] = colors[j];
                colors[j] = tmpColor;
            }
        }
    }

    // compute volume rendering
    for (int i = 0; i < tail; i++)
    {
        result += curTrancemittance * make_float3(colors[i].x, colors[i].y, colors[i].z);
        curTrancemittance *= colors[i].w;
        if (curTrancemittance <= 0.0f)
        {
            break;
        }
    }

    // update results
    const unsigned int image_index = launch_idx.y * launch_dims.x + launch_idx.x;
    const float4 accum_color = make_float4(result, curTrancemittance);
    if (subframe_index)
    {
        params.accum_buffer[image_index] += accum_color;
    }
    else
    {
        params.accum_buffer[image_index] = accum_color;
    }
    const float scale = 1.0f / (subframe_index + 1);
    params.frame_buffer[image_index] = make_color(scale * params.accum_buffer[image_index]);
}

// ----------------------------------------------------------------------------
// Miss programs
// ----------------------------------------------------------------------------
extern "C" __global__ void __miss__anyhit()
{
    float4* colorPrd = GetColorPayload();
    uint32_t* tailPrd = GetTailPayload();
    colorPrd[*tailPrd].x = params.miss_color.x;
    colorPrd[*tailPrd].y = params.miss_color.y;
    colorPrd[*tailPrd].z = params.miss_color.z;
    colorPrd[*tailPrd].w = 0.0f;
    float* TPrd = GetTPayload();
    TPrd[*tailPrd] = FLT_MAX;
    (*tailPrd)++;
}

extern "C" __global__ void __miss__closest()
{
}

// ----------------------------------------------------------------------------
// Closest hit programs
// ----------------------------------------------------------------------------
extern "C" __global__ void __closesthit__()
{
    float* tPrd = GetTPayload();
    tPrd[0] = optixGetRayTmax();
}

// ----------------------------------------------------------------------------
// Any hit scene programs
// ----------------------------------------------------------------------------
extern "C" __global__ void __intersection__()
{
    const auto* sbt_data = reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());
    const float3 ray_orig = optixGetObjectRayOrigin();
    const float3 ray_dir = optixGetObjectRayDirection();

    float tStart = optixGetRayTmin();
    float tEnd = optixGetRayTmax();
    float hStart = -1.1f;
    float hEnd = 1.1f;

    float3 pos[3];
    float3 normal[3];
    for (int i = 0; i < 3; i++)
    {
        pos[i] = make_float3(0);
        pos[i] = sbt_data->geometry_data.getSquareMesh().positions[i];
        normal[i] = make_float3(0);
        normal[i] = sbt_data->geometry_data.getSquareMesh().normals[i];
    }

    if(LocatePrismIntersection(pos, normal, ray_orig, ray_dir, tStart, tEnd, hStart, hEnd))
    {
        optixReportIntersection(tStart, 0, __float_as_int(hStart), __float_as_int(hEnd), __float_as_int(tEnd), 0);
    }

    // another prism
    pos[1] = sbt_data->geometry_data.getSquareMesh().positions[3];
    normal[1] = sbt_data->geometry_data.getSquareMesh().normals[3];
    if(LocatePrismIntersection(pos, normal, ray_orig, ray_dir, tStart, tEnd, hStart, hEnd))
    {
        optixReportIntersection(tStart, 0, __float_as_int(hStart), __float_as_int(hEnd), __float_as_int(tEnd), 1);
    }
}

extern "C" __global__ void __anyhit__()
{
    const HitGroupData* sbt_data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    uint32_t* tailPrd = GetTailPayload();
    float hIn = __int_as_float(optixGetAttribute_0());
    float hOut = __int_as_float(optixGetAttribute_1());
    float tIn = optixGetRayTmax();
    float tOut = __int_as_float(optixGetAttribute_2());
    int prismIdx = optixGetAttribute_3();

    float3 pos[3];
    float3 normals[3];
    float2 texCoords[3];
    for (int i = 0; i < 3; i++)
    {
        pos[i] = make_float3(0);
        pos[i] = sbt_data->geometry_data.getSquareMesh().positions[i];
        normals[i] = make_float3(0);
        normals[i] = sbt_data->geometry_data.getSquareMesh().normals[i];
        texCoords[i] = make_float2(0);
        texCoords[i] = sbt_data->geometry_data.getSquareMesh().texCoords[i];
    }
    if (prismIdx == 1)
    {
        pos[1] = sbt_data->geometry_data.getSquareMesh().positions[3];
        normals[1] = sbt_data->geometry_data.getSquareMesh().normals[3];
        texCoords[1] = sbt_data->geometry_data.getSquareMesh().texCoords[3];
    }

    float4* colorPrd = GetColorPayload();
    float* tPrd = GetTPayload();
    float3 baseColor = make_float3(1.0f, 1.0f, 1.0f);

    // base polygon color
    if (hIn < 0.01f)
    {
        colorPrd[*tailPrd].x = baseColor.x;
        colorPrd[*tailPrd].y = baseColor.y;
        colorPrd[*tailPrd].z = baseColor.z;
        colorPrd[*tailPrd].w = 0.0f;
        tPrd[*tailPrd] = tIn;
        (*tailPrd)++;
    }

    // prism color
    const auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>(sbt_data->volume_data[50].volume.grid);
    const auto& tree = grid->tree();
    auto acc = tree.getAccessor();
    const float opacity = sbt_data->volume_material_data.volume.opacity;
    float4 rgbt = transmittanceHDDA(hIn, hOut, grid, acc, grid, acc, opacity, ray_orig, ray_dir, tIn, tOut, pos, normals, prismIdx);

    colorPrd[*tailPrd].x = rgbt.x;
    colorPrd[*tailPrd].y = rgbt.y;
    colorPrd[*tailPrd].z = rgbt.z;
    colorPrd[*tailPrd].w = rgbt.w;
    tPrd[*tailPrd] = (tIn + tOut) / 2.0f;
    (*tailPrd)++;

    // base polygon color
    if (hOut < 0.01f)
    {
        colorPrd[*tailPrd].x = baseColor.x;
        colorPrd[*tailPrd].y = baseColor.y;
        colorPrd[*tailPrd].z = baseColor.z;
        colorPrd[*tailPrd].w = 0.0f;
        tPrd[*tailPrd] = tOut;
        (*tailPrd)++;
    }

    optixIgnoreIntersection();
}