#include <optix.h>
#include <sutil/vec_math.h>
#include <cuda/GeometryData.h>
#include <cuda/util.h>
#include <vector_types.h>

const unsigned int MAX_ITERATIONS = 10u;

inline __device__ void Swap(float& a, float& b)
{
    float tmp = a;
    a = b;
    b = tmp;
}

static __device__ float EvaluatePolynomial(float3 I, float3 p[3], float3 n[3], float h)
{
    float3 s = I - p[0] - h * n[0];
    float3 t = cross(p[1] + h * n[1] - p[0] - h * n[0], p[2] + h * n[2] - p[0] - h * n[0]);
    float f = dot(s, t);
    return f;
}

static __device__ float FindRoot(float3 I, float3 p[3], float3 n[3], float start, float end)
{
    float hr = (start + end) / 2;
    for (int i = 0; i < MAX_ITERATIONS; i++)
    {
        float3 s = I - p[0] - hr * n[0];
        float3 t = cross(p[1] + hr * n[1] - p[0] - hr * n[0], p[2] + hr * n[2] - p[0] - hr * n[0]);
        float f = dot(s, t);
        if (abs(f) < 1e-6f)
        {
            break;
        }

        // update h using Newton's method
        float u = dot(-n[0], t);
        float3 v = cross(n[1] - n[0], p[2] + hr * n[2] - p[0] - hr * n[0]) + cross(p[1] + hr * n[1] - p[0] - hr * n[0], n[2] - n[0]);
        float df = u + dot(s, v);
        float hn = hr - f / df;
        hn = fmax(start, fmin(end, hn));
        hr = hn;
    }
    return hr;
}

static __device__ float SolveEquation(float3 I, float3 p[3], float3 n[3])
{
    float hStart = -0.01f;
    float hEnd = 1.01f;

    // https://github.com/pboyer/nomial/blob/main/src/index.ts
    float rootsOfDerivative[3];
    int num = 0;
    for (int i = 0; i < 3; i++)
    {
        rootsOfDerivative[i] = 0;
    }
    float a1 = dot(-n[0], cross(n[1] - n[0], n[2] - n[0])); // hh
    float b1 = dot(-n[0], cross(n[1] - n[0], p[2] - p[0]) + cross(p[1] - p[0], n[2] - n[0])); // h
    float c1 = dot(-n[0], cross(p[1] - p[0], p[2] - p[0]));
    float a2 = dot(-n[0], cross(n[1] - n[0], n[2] - n[0]) + cross(n[1] - n[0], n[2] - n[0])); // hh
    float b2 = dot(I - p[0], cross(n[1] - n[0], n[2] - n[0]) + cross(n[1] - n[0], n[2] - n[0])) + dot(-n[0], cross(n[1] - n[0], p[2] - p[0]) + cross(p[1] - p[0], n[2] - n[0])); // h
    float c2 = dot(I - p[0], cross(n[1] - n[0], p[2] - p[0]) + cross(p[1] - p[0], n[2] - n[0]));
    float a = a1 + a2;
    float b = b1 + b2;
    float c = c1 + c2;
    float discriminant = b * b - 4 * a * c;
    if (discriminant >= 0)
    {
        const float delta = sqrt(discriminant);
        int sign = -1;
        if (b > 0)
        {
            sign = 1;
        }
        const float q = -0.5 * (b + sign * delta);
        const float x2 = q / a;
        const float x1 = c / q;
        float aa = fmin(x1, x2);
        float bb = fmax(x1, x2);
        if (hStart <= aa && aa <= hEnd)
        {
            rootsOfDerivative[num] = aa;
            num++;
        }
        if (hStart <= bb && bb <= hEnd)
        {
            rootsOfDerivative[num] = bb;
            num++;
        }
    }
    rootsOfDerivative[num] = hEnd;
    num++;

    float intervalStart = hStart;
    float fa = EvaluatePolynomial(I, p, n, hStart);
    for (int i = 0; i < num; i++)
    {
        float fb = EvaluatePolynomial(I, p, n, rootsOfDerivative[i]);
        float sign = fa * fb;
        if (sign < 0)
        {
            // a solution exists in this range
            float h = FindRoot(I, p, n, intervalStart, rootsOfDerivative[i]);
            return h;
        }
        intervalStart = rootsOfDerivative[i];
        fa = fb;
    }
    printf("solution does not exist\n");
    return -1.0f;
}

static __device__  bool IsInsideTriangle(float3 judgedTriangle[3], float3 I, float3 n)
{

    if ((dot(cross(judgedTriangle[1] - judgedTriangle[0], I - judgedTriangle[0]), n) >= 0.0f &&
        dot(cross(judgedTriangle[2] - judgedTriangle[1], I - judgedTriangle[1]), n) >= 0.0f &&
        dot(cross(judgedTriangle[0] - judgedTriangle[2], I - judgedTriangle[2]), n) >= 0.0f) ||
        (dot(cross(judgedTriangle[1] - judgedTriangle[0], I - judgedTriangle[0]), n) <= 0.0f &&
            dot(cross(judgedTriangle[2] - judgedTriangle[1], I - judgedTriangle[1]), n) <= 0.0f &&
            dot(cross(judgedTriangle[0] - judgedTriangle[2], I - judgedTriangle[2]), n) <= 0.0f))
    {
        return true;
    }
    return false;
}

static __device__  bool LocateTriangleIntersection(float3 judgedTriangle[3], float3 baseTriangle[3], float3 normals[3], float3 ray_orig, float3 ray_dir, bool& inIntersect, bool& outIntersect, float& tIn, float& tOut, float& hIn, float& hOut)
{
    float3 n, I;
    float t;
    n = normalize(cross(judgedTriangle[1] - judgedTriangle[0], judgedTriangle[2] - judgedTriangle[0]));
    t = dot(judgedTriangle[0] - ray_orig, n) / dot(ray_dir, n);
    I = ray_orig + t * ray_dir;
    if (IsInsideTriangle(judgedTriangle, I, n))
    {
        if (!inIntersect)
        {
            tIn = t;
            inIntersect = true;
            hIn = SolveEquation(I, baseTriangle, normals);
        }
        else
        {
            tOut = t;
            hOut = SolveEquation(I, baseTriangle, normals);
            if (tOut < tIn)
            {
                Swap(tIn, tOut);
                Swap(hIn, hOut);
            }
            outIntersect = true;
        }
    }
}

static __device__ bool InRange(float param, float range_min, float range_max)
{
    return (range_min <= param && param <= range_max);
}

static __device__  void LocateCoolPatchIntersection(
    float3 pos[4], float3 ray_orig, float3 ray_dir, bool& inIntersect, bool& outIntersect, float& tIn, float& tOut, float& hIn, float& hOut)
{
    // "Cool Patches: A Geometric Approach to Ray/Bilinear Patch Intersections"
    // This function was written by Alexander Reshetov and modified by Shinji Ogaki.
    // https://github.com/shinjiogaki/nonlinear-ray-tracing/blob/main/src/shell.h

    // 01 ----------- 11
    // |               |
    // | e00       e11 |
    // |      e10      |
    // 00 ----------- 10

    float3 Q00 = pos[0];
    float3 q01 = pos[2];
    float3 Q10 = pos[1];
    float3 q11 = pos[3];

    float3 e10 = Q10 - Q00;
    float3 e11 = q11 - Q10;
    float3 e00 = q01 - Q00;

    float3 q00 = Q00 - ray_orig;
    float3 q10 = Q10 - ray_orig;

    // a + b u + c u^2
    float a = dot(cross(q00, ray_dir), e00);
    float b = dot(cross(q10, ray_dir), e11);
    float c = dot(cross(q01 - q11, ray_dir), e10);

    b -= a + c;
    float det = b * b - 4 * a * c;
    if (0 > det)
    {
        return;
    }

    // Solve for u
    float u1, u2;
    if (c == 0) // trapezoid
    {
        u1 = -a / b;
        u2 = -1;
    }
    else
    {
        float content = sqrt(det) * (b >= 0 ? 1 : -1);
        u1 = (-b - content) / 2;
        u2 = a / u1;
        u1 /= c;
    }

    if (InRange(u1, 0, 1))
    {
        float3 pa = (1 - u1) * q00 + u1 * q10;
        float3 pb = (1 - u1) * e00 + u1 * e11;
        float3 n = cross(ray_dir, pb);
        float n2 = dot(n, n);
        n = cross(n, pa);
        float v1 = dot(n, ray_dir);
        if (0 <= v1 && v1 <= n2)
        {
            float h = v1 / n2;
            if (!inIntersect)
            {
                tIn = dot(n, pb) / n2;
                hIn = h;
                inIntersect = true;
            }
            else
            {
                tOut = dot(n, pb) / n2;
                hOut = h;
                if (tOut < tIn)
                {
                    Swap(tIn, tOut);
                    Swap(hIn, hOut);
                }

                outIntersect = true;
            }
        }
    }

    if (InRange(u2, 0, 1))
    {
        float3 pa = (1 - u2) * q00 + u2 * q10;
        float3 pb = (1 - u2) * e00 + u2 * e11;
        float3 n = cross(ray_dir, pb);
        float n2 = dot(n, n);
        n = cross(n, pa);
        float v2 = dot(n, ray_dir);
        if (0 <= v2 && v2 <= n2)
        {
            float h = v2 / n2;
            if (!inIntersect)
            {
                tIn = dot(n, pb) / n2;
                hIn = h;
                inIntersect = true;
            }
            else
            {
                tOut = dot(n, pb) / n2;
                hOut = h;
                if (tOut < tIn)
                {
                    Swap(tIn, tOut);
                    Swap(hIn, hOut);
                }
                outIntersect = true;
            }
        }
    }
}

static __device__ bool LocatePrismIntersection(float3 basePos[3], float3 normals[3], float3 ray_orig, float3 ray_dir, float& tIn, float& tOut, float& hIn, float& hOut)
{
    float3 offsetPos[3];
    for (int i = 0; i < 3; i++)
    {
        offsetPos[i] = make_float3(0.0f);
        offsetPos[i] = basePos[i] + normals[i];
    }
    bool inIntersect = false;
    bool outIntersect = false;
    float3 judgedTriangle[3];
    float3 judgedPatch[4];
    for (int i = 0; i < 3; i++)
    {
        judgedTriangle[i] = make_float3(0.0f);
    }

    // intersect judge with base triangle
    if (!outIntersect)
    {
        judgedTriangle[0] = basePos[0];
        judgedTriangle[1] = basePos[1];
        judgedTriangle[2] = basePos[2];
        LocateTriangleIntersection(judgedTriangle, basePos, normals, ray_orig, ray_dir, inIntersect, outIntersect, tIn, tOut, hIn, hOut);
    }

    // intersect judge with offset triangle
    if (!outIntersect)
    {
        judgedTriangle[0] = offsetPos[0];
        judgedTriangle[1] = offsetPos[1];
        judgedTriangle[2] = offsetPos[2];
        LocateTriangleIntersection(judgedTriangle, basePos, normals, ray_orig, ray_dir, inIntersect, outIntersect, tIn, tOut, hIn, hOut);
    }

    // intersect judge with rectangle p0 p1 p0Offset p1Offset
    if (!outIntersect)
    {
        judgedPatch[0] = basePos[0];
        judgedPatch[1] = basePos[1];
        judgedPatch[2] = offsetPos[0];
        judgedPatch[3] = offsetPos[1];
        LocateCoolPatchIntersection(judgedPatch, ray_orig, ray_dir, inIntersect, outIntersect, tIn, tOut, hIn, hOut);
    }

    // intersect judge with rectangle p1 p2 p1Offset p2Offset
    if (!outIntersect)
    {
        judgedPatch[0] = basePos[1];
        judgedPatch[1] = basePos[2];
        judgedPatch[2] = offsetPos[1];
        judgedPatch[3] = offsetPos[2];
        LocateCoolPatchIntersection(judgedPatch, ray_orig, ray_dir, inIntersect, outIntersect, tIn, tOut, hIn, hOut);
    }

    // intersect judge with rectangle p2 p0 p2Offset p0Offset
    if (!outIntersect)
    {
        judgedPatch[0] = basePos[2];
        judgedPatch[1] = basePos[0];
        judgedPatch[2] = offsetPos[2];
        judgedPatch[3] = offsetPos[0];
        LocateCoolPatchIntersection(judgedPatch, ray_orig, ray_dir, inIntersect, outIntersect, tIn, tOut, hIn, hOut);
    }

    return outIntersect;
}