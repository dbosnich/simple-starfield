//--------------------------------------------------------------
// Copyright (c) David Bosnich <david.bosnich.public@gmail.com>
//
// This code is licensed under the MIT License, a copy of which
// can be found in the license.txt file included at the root of
// this distribution, or at https://opensource.org/licenses/MIT
//--------------------------------------------------------------

#include <memory_utils.h>
#include <starfield.h>
#include <inttypes.h>
#include <assert.h>
#include <cstring>
#include <random>
#include <new>

#ifdef CUDA_SUPPORTED
#   include <cuda_runtime.h>
#endif // CUDA_SUPPORTED

using namespace Simple;
using namespace Simple::Display;

//--------------------------------------------------------------
Starfield::Starfield(const Config& a_config)
    : m_config(a_config)
    , m_starsBuffer(nullptr)
    , m_context(nullptr)
{
}

//--------------------------------------------------------------
void Starfield::StartUp()
{
    assert(m_context == nullptr);
    m_context = new Context(m_config.displayContextConfig);

    const Buffer& buffer = m_context->GetBuffer();
    const Buffer::Interop bufferInterop = buffer.GetInterop();
    const size_t sizeBytes = m_config.numStars * sizeof(Star);
    if (bufferInterop == Buffer::Interop::HOST)
    {
        void* starsBuffer = malloc(sizeBytes);
        memset(starsBuffer, 0, sizeBytes);
        m_starsBuffer = static_cast<Star*>(starsBuffer);
    }
    else if (bufferInterop == Buffer::Interop::CUDA)
    {
    #ifdef CUDA_SUPPORTED
        cudaMalloc(&m_starsBuffer, sizeBytes);
        cudaMemset(m_starsBuffer, 0, sizeBytes);
    #endif // CUDA_SUPPORTED
    }

    // Shut down immediately if there is no display buffer.
    if (!m_context->GetBuffer().GetData())
    {
        RequestShutDown();
    }
}

//--------------------------------------------------------------
void Starfield::ShutDown()
{
    const Buffer& buffer = m_context->GetBuffer();
    const Buffer::Interop bufferInterop = buffer.GetInterop();
    if (bufferInterop == Buffer::Interop::HOST)
    {
        free(m_starsBuffer);
    }
    else if (bufferInterop == Buffer::Interop::CUDA)
    {
    #ifdef CUDA_SUPPORTED
        cudaFree(m_starsBuffer);
    #endif // CUDA_SUPPORTED
    }
    m_starsBuffer = nullptr;

    delete m_context;
    m_context = nullptr;
}

//--------------------------------------------------------------
void Starfield::UpdateStart(float a_deltaTimeSeconds)
{
    (void)a_deltaTimeSeconds;
    m_context->OnFrameStart();
}

//--------------------------------------------------------------
void Starfield::UpdateFixed(float a_fixedTimeSeconds)
{
    if (m_context->GetWindow() &&
        m_context->GetWindow()->IsClosed())
    {
        RequestShutDown();
        return;
    }

    switch (m_context->GetBuffer().GetFormat())
    {
        case Buffer::Format::RGBA_FLOAT:
        {
            UpdateStars<float>(a_fixedTimeSeconds);
        }
        break;
        case Buffer::Format::RGBA_UINT8:
        {
            UpdateStars<uint8_t>(a_fixedTimeSeconds);
        }
        break;
        case Buffer::Format::RGBA_UINT16:
        {
            UpdateStars<uint16_t>(a_fixedTimeSeconds);
        }
        break;
        default:
        {
        }
        break;
    }
}

//--------------------------------------------------------------
void Starfield::UpdateEnded(float a_deltaTimeSeconds)
{
    (void)a_deltaTimeSeconds;
    m_context->OnFrameEnded();
}

//--------------------------------------------------------------
#ifdef CUDA_SUPPORTED
template <typename DataType>
extern void UpdateStarsCuda(const Starfield::Config& a_config,
                            const Display::Context& a_context,
                            Starfield::Star* a_stars,
                            float a_secondsElapsed);
#endif // CUDA_SUPPORTED

//--------------------------------------------------------------
float RandomFloat(float a_min, float a_max)
{
    static std::random_device s_randomDevice;
    static std::mt19937 s_generator(s_randomDevice());
    std::uniform_real_distribution<float> distribution(a_min, a_max);
    return distribution(s_generator);
}

//--------------------------------------------------------------
template<typename DataType>
DataType ChannelValueFromFloat(float a_value);

//--------------------------------------------------------------
template<> float ChannelValueFromFloat<float>(float a_value)
{
    return std::max(0.0f, std::min(a_value, 1.0f));
}

//--------------------------------------------------------------
template<> uint8_t ChannelValueFromFloat<uint8_t>(float a_value)
{
    constexpr float UINT8_MAX_AS_FLOAT = static_cast<float>(UINT8_MAX);
    a_value = std::max(0.0f, std::min(a_value, 1.0f));
    a_value *= UINT8_MAX_AS_FLOAT;
    return static_cast<uint8_t>(a_value);
}

//--------------------------------------------------------------
template<> uint16_t ChannelValueFromFloat<uint16_t>(float a_value)
{
    constexpr float UINT16_MAX_AS_FLOAT = static_cast<float>(UINT16_MAX);
    a_value = std::max(0.0f, std::min(a_value, 1.0f));
    a_value *= UINT16_MAX_AS_FLOAT;
    return static_cast<uint16_t>(a_value);
}

//--------------------------------------------------------------
template<typename DataType>
DataType ChannelValueFromIndex(const Starfield::Star& a_star,
                               uint32_t a_index)
{
    switch (a_index)
    {
        case 0: return ChannelValueFromFloat<DataType>(a_star.r);
        case 1: return ChannelValueFromFloat<DataType>(a_star.g);
        case 2: return ChannelValueFromFloat<DataType>(a_star.b);
        case 3: return ChannelValueFromFloat<DataType>(1.0f); // Alpha
        default: return ChannelValueFromFloat<DataType>(0.0f);
    }
}

//--------------------------------------------------------------
template<typename DataType>
void UpdateStarsHost(const Starfield::Config& a_config,
                     Display::Context& a_context,
                     Starfield::Star* a_stars,
                     float a_secondsElapsed)
{
    // Cache buffer values.
    const Buffer& buffer = a_context.GetBuffer();
    const uint32_t bufferSize = buffer.GetSize();
    const uint32_t bufferWidth = buffer.GetWidth();
    const uint32_t bufferHeight = buffer.GetHeight();
    DataType* bufferData = buffer.GetData<DataType>();
    const uint32_t numChannels = buffer.ChannelsPerPixel(buffer.GetFormat());
    if (!bufferSize || !bufferWidth || !bufferHeight || !bufferData || !numChannels)
    {
        return;
    }

    // Clear the display buffer.
    memset(bufferData, 0, bufferSize);

    // Draw the stars.
    const float sceneWidth = static_cast<float>(bufferWidth);
    const float sceneHeight = static_cast<float>(bufferHeight);
    const float sceneCenterX = sceneWidth * 0.5f;
    const float sceneCenterY = sceneHeight * 0.5f;
    const uint32_t numStars = a_config.numStars;
    for (uint32_t i = 0; i < numStars; ++i)
    {
        Starfield::Star& starInstance = a_stars[i];
        if (starInstance.z <= a_config.zNear)
        {
            // The star has passed the near clip plane,
            // position it randomly in the visible range.
            starInstance.x = (RandomFloat(0.0f, 1.0f) * sceneWidth) - sceneCenterX;
            starInstance.y = (RandomFloat(0.0f, 1.0f) * sceneHeight) - sceneCenterY;
            starInstance.z = (RandomFloat(0.0f, 1.0f) * (a_config.zFar - a_config.zNear));

            // Recolor star
            starInstance.r = RandomFloat(0.0f, 1.0f);
            starInstance.g = RandomFloat(0.0f, 1.0f);
            starInstance.b = RandomFloat(0.0f, 1.0f);
        }
        else
        {
            // The star is still visible,
            // move it towards the camera.
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
        for (int32_t x = (int32_t)posX; x < (int32_t)(posX + width); ++x)
        {
            if (x < 0 || (uint32_t)x >= bufferWidth)
            {
                continue;
            }
            for (int32_t y = (int32_t)posY; y < (int32_t)(posY + height); ++y)
            {
                if (y < 0 || (uint32_t)y >= bufferHeight)
                {
                    continue;
                }
                const uint32_t index = ((x * numChannels) +
                                        (y * bufferWidth * numChannels));
                for (uint32_t z = 0; z < numChannels; ++z)
                {
                    bufferData[index + z] = ChannelValueFromIndex<DataType>(starInstance, z);
                }
            }
        }
    }
}

//--------------------------------------------------------------
template<typename DataType>
void Starfield::UpdateStars(float a_secondsElapsed)
{
    const Buffer& buffer = m_context->GetBuffer();
    if (buffer.GetInterop() == Buffer::Interop::HOST)
    {
        UpdateStarsHost<DataType>(m_config,
                                  *m_context,
                                  m_starsBuffer,
                                  a_secondsElapsed);
    }
    else if (buffer.GetInterop() == Buffer::Interop::CUDA)
    {
    #ifdef CUDA_SUPPORTED
        UpdateStarsCuda<DataType>(m_config,
                                  *m_context,
                                  m_starsBuffer,
                                  a_secondsElapsed);
    #endif // CUDA_SUPPORTED
    }
}

//--------------------------------------------------------------
void Starfield::OnFrameComplete(const FrameStats& a_stats)
{
    (void)a_stats;
#ifndef NDEBUG
    auto toMs = [](const Duration& a_duration)->int64_t
    {
        using namespace std::chrono;
        return duration_cast<milliseconds>(a_duration).count();
    };

    printf("Frame Count:    %" PRIu64 "\n"
           "Average FPS:    %" PRIu32 "\n"
           "Target FPS:     %" PRIu32 "\n"
           "Actual Dur:     %" PRIi64 " (ms)\n"
           "Target Dur:     %" PRIi64 " (ms)\n"
           "Excess Dur:     %" PRIi64 " (ms)\n"
           "Total Dur:      %" PRIi64 " (ms)\n"
           "\n",
           a_stats.frameCount,
           a_stats.averageFPS,
           a_stats.targetFPS,
           toMs(a_stats.actualDur),
           toMs(a_stats.targetDur),
           toMs(a_stats.excessDur),
           toMs(a_stats.totalDur));
#endif // NDEBUG
}
