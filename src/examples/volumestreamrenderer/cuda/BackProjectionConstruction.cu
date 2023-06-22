/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "BackProjectionConstruction.h"

#include <claraviz/hardware/cuda/CudaService.h>

#include <stdint.h>

namespace clara::viz
{

/**
  * @brief function to convert 3D cartesian coordinates to shperical coordinates.
  *
  *
  * @param[in] inX                X coordinate in 3d cartesian space.
  *
  * @param[in] inY                Y coordinate in 3d cartesian space.
  *
  * @param[in] inZ                Z coordinate in 3d cartesian space.
  *
  * @param[out] radius            Radius from center.
  *
  * @param[out] phi               Elevation angle defined from XY-plane up.
  *
  * @param[out] theta             Angle made on XY-plane.
  */

__device__ void getShpericalCoordinates(int16_t inX, int16_t inY, int16_t inZ, float *radius, float *phi, float *theta)
{
    float x     = inX;
    float y     = inY;
    float z     = inZ;
    float xy_sq = x * x + y * y;

    // Radius (lenght from center)
    *radius = sqrtf((float)(xy_sq + z * z));

    // for elevation angle defined from XY-plane up
    *phi = atan2f(z, sqrtf(xy_sq));

    // for angle made on XY-plane
    *theta = atan2f(y, x);
}

/**
  * @brief function to convert polar coordinates to cartesian coordinates.
  *
  * @param[in] radius            Radius from center.
  *
  * @param[in] theta             Angle made on XY-plane.
  *
  * @param[in] dimY              Total length in dimension Y.
  *
  * @param[out] x                X coordinate in cartesian space.
  *
  * @param[out] y                Y coordinate in cartesian space.
  */

__device__ void getPolToCart(float radius, float theta, uint16_t dimY, uint16_t *x, uint16_t *y)
{
    *x = (uint16_t)(radius * cosf(theta));
    *y = (uint16_t)(radius * sinf(theta) + dimY / 2.0f);
}

/**
  * @brief getNearestNeighborInterp gets the index of the nearest neighbor for a
  * given value and calculates the interpolation parameter using neighbor at
  * (index - 1) and (index + 1).
  *
  * @param[in] phiSorted        1D array contains sorted values of phi angle
  *                              for every 2d image in input.
  *
  * @param[in] lenOfPhi          Length of array phi and number of 2d images
  *                              present in input.
  *
  * @param[in] phiToSearch       value to search neighbor for and then use to
  *                              calculate interpolation parameter.
  *
  * @param[out] firstIdx         Index in phiSorted such that phiToSearch >=
  *                              phiSorted[firstIdx].
  *
  * @param[out] secondIndex      Index in phiSorted such that phiToSearch <=
  *                              phiSorted[secondIndex].
  *
  * @param[out] interParam       Interpolation parater.
  */

__device__ void getNearestNeighborInterp(const float *phiSorted, uint16_t lenOfPhi, float phiToSearch,
                                         uint16_t *firstIdx, uint16_t *secondIndex, float *interParam)
{
    float max = phiSorted[lenOfPhi - 1];
    float min = phiSorted[0];

    float distance = fabsf(max - min) / (float)(lenOfPhi - 1);

    // assuming on average phi will be uniformaly distributed.
    uint16_t index = (uint16_t)(fabsf(phiToSearch - min) / distance);

    if (index >= lenOfPhi)
    {
        index = lenOfPhi - 1;
    }

    if (phiToSearch > phiSorted[index])
    {
        while (index < (lenOfPhi - 1) && phiToSearch >= phiSorted[index + 1])
        {
            ++index;
        }
    }
    else if (phiToSearch < phiSorted[index])
    {
        while (index > 0 && phiToSearch < phiSorted[index])
        {
            --index;
        }
    }

    *firstIdx = index;

    // find the interpolation parameter as well
    *secondIndex = *firstIdx + 1;
    if (*secondIndex >= lenOfPhi)
        *secondIndex = lenOfPhi - 1;

    float norm = phiSorted[*secondIndex] - phiSorted[*firstIdx];
    // protection against division by zero
    if (norm <= 0.000001)
        norm = 1.0;

    *interParam = (phiToSearch - phiSorted[*firstIdx]) / norm;

    return;
}

/**
  * @brief constructVolume is a cuda kernel where for every voxel in the final
  * 3d volume we figure out which nearest two 2d frames to use to calculate the
  * interpolated value.
  *
  * @param[out] reconstruction   1D array for final 3d constructed volume.
  *
  * @param[in] input             1D array contains 2d images one after another.
  *
  * @param[in] phi               1D array contains sorted values of phi angle
  *                              for every 2d image in input.
  *
  * @param[in] dimXYZOfOutput    Depth, width and height of 3d volume, respectively.
  *
  * @param[in] dimXYOfInput      Height and width of 2d images, respectively.
  *
  * @param[in] lenOfPhi          Length of array phi and number of 2d images
  *                              present in input.
  *
  * @param[in] writePhiDataTimes How many elements one thread should write the
  *                              shared data.
  *
  * @param[in] subtractBottomLines Remove bottom solid white lines for better
  *                                visualization.
  */

__global__ void constructVolume(uint8_t *reconstruction, const uint8_t *input, const float *phiSorted,
                                ushort3 dimXYZOfOutput, ushort2 dimXYOfInput, uint16_t lenOfPhi,
                                uint16_t writePhiDataTimes, uint16_t subtractBottomLines)
{
    extern __shared__ float phidata[];

    uint idx = blockIdx.y * blockDim.x + threadIdx.x;

    // fill out the phidata with phiSorted for every thread in a block
    for (uint16_t s = 0; s < writePhiDataTimes; s++)
    {
        uint16_t index = threadIdx.x * writePhiDataTimes + s;
        if (index < lenOfPhi)
            phidata[index] = phiSorted[index];
    }

    __syncthreads();

    uint16_t inputHeight = dimXYOfInput.x;
    uint16_t inputWidth  = dimXYOfInput.y;

    uint16_t outputXDim = dimXYZOfOutput.x;
    uint16_t outputYDim = dimXYZOfOutput.y;
    uint16_t outputZDim = dimXYZOfOutput.z;

    uint16_t vX = idx / outputYDim;
    uint16_t vY = idx % outputYDim;
    uint16_t vZ = blockIdx.x;

    uint16_t yc = outputYDim / 2;
    uint16_t zc = outputZDim / 2;

    if (vX >= outputXDim || vY >= outputYDim || vZ >= outputZDim)
        return;

    float r, phi, theta = 0.0f;
    int16_t vx_xc = vX;
    int16_t vy_yc = vY - yc;
    int16_t vz_zc = vZ - zc;
    getShpericalCoordinates(vx_xc, vy_yc, vz_zc, &r, &phi, &theta);

    if ((phi < phiSorted[0]) || (phi > phiSorted[lenOfPhi - 1]))
    {
        // outside of input phi range
        reconstruction[vX * outputYDim * outputZDim + vY * outputZDim + vZ] = 0;
    }
    else
    {
        uint16_t firstIdx  = 0;
        uint16_t secondIdx = 0;
        float interParam   = 0;
        getNearestNeighborInterp(phidata, lenOfPhi, phi, &firstIdx, &secondIdx, &interParam);

        uint16_t iX = 0;
        uint16_t iY = 0;
        getPolToCart(r, theta, inputWidth, &iX, &iY);

        if (iX >= (inputHeight - subtractBottomLines) || iY >= inputWidth)
        {
            // outside of input slice
            reconstruction[vX * outputYDim * outputZDim + vY * outputZDim + vZ] = 0;
        }
        else
        {
            reconstruction[vX * outputYDim * outputZDim + vY * outputZDim + vZ] =
                input[firstIdx * inputHeight * inputWidth + iX * inputWidth + iY] * (1 - interParam) +
                input[secondIdx * inputHeight * inputWidth + iX * inputWidth + iY] * (interParam);
        }
    }
}

std::unique_ptr<CudaFunctionLauncher> GetBackProjectionLauncher(size_t shared_mem_size)
{
    return std::unique_ptr<CudaFunctionLauncher>(new CudaFunctionLauncher(&constructVolume, shared_mem_size));
}

} // namespace clara::viz