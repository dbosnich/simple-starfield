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

//--------------------------------------------------------------
template<typename DataType>
__device__ DataType ChannelValueFromFloatDevice(float a_value);

//--------------------------------------------------------------
template<>
__device__ float ChannelValueFromFloatDevice<float>(float a_value)
{
    return max(0.0f, min(a_value, 1.0f));
}

//--------------------------------------------------------------
template<>
__device__ uint8_t ChannelValueFromFloatDevice<uint8_t>(float a_value)
{
    constexpr float UINT8_MAX_AS_FLOAT = static_cast<float>(UINT8_MAX);
    a_value = max(0.0f, min(a_value, 1.0f));
    a_value *= UINT8_MAX_AS_FLOAT;
    return static_cast<uint8_t>(a_value);
}

//--------------------------------------------------------------
template<>
__device__ uint16_t ChannelValueFromFloatDevice<uint16_t>(float a_value)
{
    constexpr float UINT16_MAX_AS_FLOAT = static_cast<float>(UINT16_MAX);
    a_value = max(0.0f, min(a_value, 1.0f));
    a_value *= UINT16_MAX_AS_FLOAT;
    return static_cast<uint16_t>(a_value);
}

//--------------------------------------------------------------
template<typename DataType>
__device__ DataType ChannelValueFromIndexDevice(const Starfield::Star& a_star,
                                                uint32_t a_index)
{
    switch (a_index)
    {
        case 0: return ChannelValueFromFloatDevice<DataType>(a_star.r);
        case 1: return ChannelValueFromFloatDevice<DataType>(a_star.g);
        case 2: return ChannelValueFromFloatDevice<DataType>(a_star.b);
        case 3: return ChannelValueFromFloatDevice<DataType>(1.0f); // Alpha
        default:  return ChannelValueFromFloatDevice<DataType>(0.0f);
    }
}

//--------------------------------------------------------------
template <typename DataType>
__global__ void UpdateStarsKernel(const Starfield::Config a_config,
                                  Starfield::Star* a_stars,
                                  float a_secondsElapsed,
                                  DataType* a_bufferData,
                                  uint32_t a_bufferWidth,
                                  uint32_t a_bufferHeight,
                                  uint32_t a_numChannels,
                                  uint64_t a_randomSeed)
{
    // Calculate the kernel id.
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a_config.numStars)
    {
        return;
    }

    // Random state.
    curandState state;
    curand_init(a_randomSeed, idx, 0, &state);

    // Calculate scene values.
    const float sceneWidth = static_cast<float>(a_bufferWidth);
    const float sceneHeight = static_cast<float>(a_bufferHeight);
    const float sceneCenterX = sceneWidth * 0.5f;
    const float sceneCenterY = sceneHeight * 0.5f;

    // Update one star in each kernel.
    Starfield::Star& starInstance = a_stars[idx];
    if (starInstance.z <= a_config.zNear)
    {
        // The star has passed the near clip plane,
        // position it randomly in the visible range.
        starInstance.x = (curand_uniform(&state) * sceneWidth) - sceneCenterX;
        starInstance.y = (curand_uniform(&state) * sceneHeight) - sceneCenterY;
        starInstance.z = (curand_uniform(&state) * (a_config.zFar - a_config.zNear)) + a_config.zNear;

        // Randomly color the star.
        starInstance.r = curand_uniform(&state);
        starInstance.g = curand_uniform(&state);
        starInstance.b = curand_uniform(&state);
    }
    else
    {
        // The star is still visible, move it towards the camera.
        starInstance.z -= a_config.starSpeed * a_secondsElapsed;
    }

    // Project the star onto the screen.
    const float posX = ((starInstance.x * a_config.focalLength) / starInstance.z) + sceneCenterX;
    const float posY = ((starInstance.y * a_config.focalLength) / starInstance.z) + sceneCenterY;

    // Scale the star such that it gets bigger as it moves towards the screen.
    const float scale = (1.0f - (starInstance.z / a_config.zFar));
    const float width = (a_config.starWidth * scale);
    const float height = (a_config.starHeight * scale);

    // Draw the star to the pixel buffer.
    const uint32_t minX = static_cast<uint32_t>(max(0.0f, posX));
    const uint32_t maxX = static_cast<uint32_t>(min(sceneWidth,
                                                max(0.0f, posX + width)));
    const uint32_t minY = static_cast<uint32_t>(max(0.0f, posY));
    const uint32_t maxY = static_cast<uint32_t>(min(sceneHeight,
                                                max(0.0f, posY + height)));
    for (uint32_t x = minX; x < maxX; ++x)
    {
        for (uint32_t y = minY; y < maxY; ++y)
        {
            const uint32_t index = ((x * a_numChannels) + (y * a_bufferWidth * a_numChannels));
            for (uint32_t z = 0; z < a_numChannels; ++z)
            {
                a_bufferData[index + z] = ChannelValueFromIndexDevice<DataType>(starInstance, z);
            }
        }
    }
}

//--------------------------------------------------------------
template <typename DataType>
void UpdateStarsCuda(const Starfield::Config& a_config,
                     const Display::Context& a_context,
                     Starfield::Star* a_stars,
                     float a_secondsElapsed)
{
    const Buffer& buffer = a_context.GetBuffer();
    const uint32_t bufferSize = buffer.GetSize();
    const uint32_t bufferWidth = buffer.GetWidth();
    const uint32_t bufferHeight = buffer.GetHeight();
    DataType* bufferData = buffer.GetData<DataType, Buffer::Interop::CUDA>();
    const uint32_t numChannels = buffer.ChannelsPerPixel(buffer.GetFormat());
    if (!bufferSize || !bufferWidth || !bufferHeight || !bufferData || !numChannels)
    {
        return;
    }

    // Clear the display buffer.
    cudaMemset(bufferData, 0, bufferSize);

    // Prepare for kernel launch.
    const uint32_t threadsPerBlock = 256;
    const uint32_t numBlocks = (a_config.numStars + threadsPerBlock - 1) / threadsPerBlock;
    const uint64_t randomSeed = static_cast<uint64_t>(time(nullptr));
    UpdateStarsKernel<DataType><<<numBlocks, threadsPerBlock>>>(a_config,
                                                                a_stars,
                                                                a_secondsElapsed,
                                                                bufferData,
                                                                bufferWidth,
                                                                bufferHeight,
                                                                numChannels,
                                                                randomSeed);
}

//--------------------------------------------------------------
template void UpdateStarsCuda<float>(const Starfield::Config& a_config,
                                     const Display::Context& a_context,
                                     Starfield::Star* a_stars,
                                     float a_secondsElapsed);

//--------------------------------------------------------------
template void UpdateStarsCuda<uint8_t>(const Starfield::Config& a_config,
                                       const Display::Context& a_context,
                                       Starfield::Star* a_stars,
                                       float a_secondsElapsed);

//--------------------------------------------------------------
template void UpdateStarsCuda<uint16_t>(const Starfield::Config& a_config,
                                        const Display::Context& a_context,
                                        Starfield::Star* a_stars,
                                        float a_secondsElapsed);
