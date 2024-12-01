
#include <cuda_runtime.h>
#include "nanovdb/NanoVDB.h"
#include <optix.h>

class HDDA
{
public:
    enum Axis { u, v, w, all };

    __device__ HDDA(const nanovdb::Vec3f uvw, const nanovdb::Vec3f duvw, int dim)
    {
        Update(uvw, duvw, dim);
    }

    __device__ bool Update(const nanovdb::Vec3f uvw, const nanovdb::Vec3f duvw, int dim)
    {
        if (dim == _dim)
        {
            return false;
        }
        nanovdb::Coord ijk = nanovdb::RoundDown<nanovdb::Coord>(uvw);
        nanovdb::Coord voxel = ijk & ~(dim - 1);
        _dim = dim;
        _axis = 3;
        for (int i = 0; i < 3; i++)
        {
            if (duvw[i] < 0.0f)
            {
                _paramCandidate[i] = voxel[i] - 0.01f;
                _step[i] = -1.0f;
            }
            else
            {
                _paramCandidate[i] = voxel[i] + dim + 0.01f;
                _step[i] = 1.0f;
            }
        }
        return true;
    }

    __device__ bool Step(
        nanovdb::Vec3f& uvwCurrent,
        const nanovdb::Vec3f uvwCandidateU,
        const nanovdb::Vec3f uvwCandidateV,
        const nanovdb::Vec3f uvwCandidateW,
        float& hCurrent, float hU, float hV, float hW,
        float& alphaCurrent, float alphaU, float alphaV, float alphaW,
        float& betaCurrent, float betaU, float betaV, float betaW)
    {

        nanovdb::Vec3f diffCandidate;
        diffCandidate[0] = (uvwCandidateU - uvwCurrent).length();
        diffCandidate[1] = (uvwCandidateV - uvwCurrent).length();
        diffCandidate[2] = (uvwCandidateW - uvwCurrent).length();
        _axis = MinIndex(diffCandidate);

        if (_axis == u) 
        {
            uvwCurrent = uvwCandidateU;
            hCurrent = hU;
            alphaCurrent = alphaU;
            betaCurrent = betaU;
        }
        else if (_axis == v)
        {
            uvwCurrent = uvwCandidateV;
            hCurrent = hV;
            alphaCurrent = alphaV;
            betaCurrent = betaV;
        }
        else if (_axis == w)
        {
            uvwCurrent = uvwCandidateW;
            hCurrent = hW;
            alphaCurrent = alphaW;
            betaCurrent = betaW;
        }
        _paramCandidate[_axis] += _dim * _step[_axis];
    }

    __device__ float GetParamCandidate(int axis) const 
    {
        return _paramCandidate[axis]; 
    }

    __device__ int GetAxis() const 
    { 
        return _axis; 
    }

private:
    int _dim = 0;
    int _axis = 3;
    nanovdb::Vec3f _paramCandidate;
    nanovdb::Vec3f _step = nanovdb::Vec3f(1.0f, 1.0f, 1.0f);
};
