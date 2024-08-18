//--------------------------------------------------------------
// Copyright (c) David Bosnich <david.bosnich.public@gmail.com>
//
// This code is licensed under the MIT License, a copy of which
// can be found in the license.txt file included at the root of
// this distribution, or at https://opensource.org/licenses/MIT
//--------------------------------------------------------------

#include <simple/display/buffer.h>
#include <starfield.h>

#include <curand_kernel.h>

using namespace Simple::Display;
using namespace Simple;
using namespace std;

using Stars = vector<Starfield::Star>;

template <typename DataType, uint32_t ChannelsPerPixel>
__constant__ DataType BACK_COLOR[ChannelsPerPixel];

template <typename DataType, uint32_t ChannelsPerPixel>
__constant__ DataType STAR_COLOR[ChannelsPerPixel];

//--------------------------------------------------------------
template <typename DataType, uint32_t ChannelsPerPixel>
__global__ void SetBackgroundKernel(DataType* a_bufferData,
                                    uint32_t a_bufferWidth,
                                    uint32_t a_bufferHeight)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= a_bufferWidth || y >= a_bufferHeight)
    {
        return;
    }

    const uint32_t i = (x * ChannelsPerPixel) +
                       (y * a_bufferWidth * ChannelsPerPixel);
    for (uint32_t z = 0; z < ChannelsPerPixel; ++z)
    {
        a_bufferData[i + z] = BACK_COLOR<DataType, ChannelsPerPixel>[z];
    }
}

//--------------------------------------------------------------
template <typename DataType, uint32_t ChannelsPerPixel>
__global__ void UpdateStarsKernel(const Starfield::Config a_config,
                                  Starfield::Star* a_stars,
                                  size_t a_numDeviceStars,
                                  float a_secondsElapsed,
                                  DataType* a_bufferData,
                                  uint32_t a_bufferWidth,
                                  uint32_t a_bufferHeight,
                                  uint64_t a_randomSeed)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a_numDeviceStars)
    {
        return;
    }

    // Update and draw stars
    curandState state;
    curand_init(a_randomSeed, idx, 0, &state);

    const float sceneWidth = static_cast<float>(a_bufferWidth);
    const float sceneHeight = static_cast<float>(a_bufferHeight);
    const float sceneCenterX = sceneWidth * 0.5f;
    const float sceneCenterY = sceneHeight * 0.5f;

    Starfield::Star& starInstance = a_stars[idx];
    if (starInstance.z <= a_config.zNear)
    {
        // Reposition star
        starInstance.x = (curand_uniform(&state) * sceneWidth) - sceneCenterX;
        starInstance.y = (curand_uniform(&state) * sceneHeight) - sceneCenterY;
        starInstance.z = (curand_uniform(&state) * (a_config.zFar - a_config.zNear)) + a_config.zNear;
    }
    else
    {
        // Move star towards camera
        starInstance.z -= a_config.starSpeed * a_secondsElapsed;
    }

    // Project star onto screen
    const float posX = ((starInstance.x * a_config.focalLength) / starInstance.z) + sceneCenterX;
    const float posY = ((starInstance.y * a_config.focalLength) / starInstance.z) + sceneCenterY;

    // Scale star
    const float scale = (1.0f - (starInstance.z / a_config.zFar));
    const float width = (a_config.starWidth * scale);
    const float height = (a_config.starHeight * scale);

    // Draw star
    for (int32_t x = (int32_t)posX; x < (int32_t)(posX + width); ++x)
    {
        if (x < 0 || (uint32_t)x >= a_bufferWidth) continue;
        for (int32_t y = (int32_t)posY; y < (int32_t)(posY + height); ++y)
        {
            if (y < 0 || (uint32_t)y >= a_bufferHeight) continue;
            const uint32_t index = ((x * ChannelsPerPixel) + (y * a_bufferWidth * ChannelsPerPixel));
            for (uint32_t z = 0; z < ChannelsPerPixel; ++z)
            {
                a_bufferData[index + z] = STAR_COLOR<DataType, ChannelsPerPixel>[z];
            }
        }
    }
}

//--------------------------------------------------------------
template <typename DataType, uint32_t ChannelsPerPixel>
void UpdateStarsCuda(const Starfield::Config& a_config,
                     const Display::Context& a_context,
                     Starfield::Star* a_stars,
                     size_t a_numDeviceStars,
                     float a_secondsElapsed)
{
    const Buffer& buffer = a_context.GetBuffer();
    const uint32_t bufferWidth = buffer.GetWidth();
    const uint32_t bufferHeight = buffer.GetHeight();
    DataType* bufferData = buffer.GetData<DataType, Buffer::Interop::CUDA>();
    if (!bufferWidth || !bufferHeight || !bufferData)
    {
        return;
    }

    // Clear the display buffer.
    const dim3 blockDim(16, 16);
    const dim3 gridDim((bufferWidth + blockDim.x - 1) / blockDim.x,
                       (bufferHeight + blockDim.y - 1) / blockDim.y);
    SetBackgroundKernel<DataType, ChannelsPerPixel><<<gridDim, blockDim>>>(bufferData,
                                                                           bufferWidth,
                                                                           bufferHeight);

    // Prepare for kernel launch.
    const uint32_t threadsPerBlock = 256;
    const uint32_t numBlocks = ((uint32_t)a_numDeviceStars + threadsPerBlock - 1) / threadsPerBlock;
    const uint64_t randomSeed = static_cast<uint64_t>(time(nullptr));
    UpdateStarsKernel<DataType, ChannelsPerPixel><<<numBlocks, threadsPerBlock>>>(a_config,
                                                                                  a_stars,
                                                                                  a_numDeviceStars,
                                                                                  a_secondsElapsed,
                                                                                  bufferData,
                                                                                  bufferWidth,
                                                                                  bufferHeight,
                                                                                  randomSeed);
}

//--------------------------------------------------------------
template <typename DataType, uint32_t ChannelsPerPixel>
void UpdateStarsCuda(const Starfield::Config& a_config,
                     const Display::Context& a_context,
                     const DataType a_backColor[ChannelsPerPixel],
                     const DataType a_starColor[ChannelsPerPixel],
                     const Stars& a_hostStars,
                     Starfield::Star* a_stars,
                     float a_secondsElapsed)
{
    cudaMemcpyToSymbol(BACK_COLOR<DataType, ChannelsPerPixel>, a_backColor, ChannelsPerPixel * sizeof(DataType));
    cudaMemcpyToSymbol(STAR_COLOR<DataType, ChannelsPerPixel>, a_starColor, ChannelsPerPixel * sizeof(DataType));
    UpdateStarsCuda<DataType, ChannelsPerPixel>(a_config,
                                                a_context,
                                                a_stars,
                                                a_hostStars.size(),
                                                a_secondsElapsed);
}

//--------------------------------------------------------------
template void UpdateStarsCuda<float, 4>(const Starfield::Config& a_config,
                                        const Display::Context& a_context,
                                        const float a_backColor[4],
                                        const float a_starColor[4],
                                        const Stars& a_hostStars,
                                        Starfield::Star* a_stars,
                                        float a_secondsElapsed);

//--------------------------------------------------------------
template void UpdateStarsCuda<uint8_t, 4>(const Starfield::Config& a_config,
                                          const Display::Context& a_context,
                                          const uint8_t a_backColor[4],
                                          const uint8_t a_starColor[4],
                                          const Stars& a_hostStars,
                                          Starfield::Star* a_stars,
                                          float a_secondsElapsed);

//--------------------------------------------------------------
template void UpdateStarsCuda<uint16_t, 4>(const Starfield::Config& a_config,
                                           const Display::Context& a_context,
                                           const uint16_t a_backColor[4],
                                           const uint16_t a_starColor[4],
                                           const Stars& a_hostStars,
                                           Starfield::Star* a_stars,
                                           float a_secondsElapsed);
