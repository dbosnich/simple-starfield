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
#include <random>

#ifdef CUDA_SUPPORTED
#   include <cuda_runtime.h>
#endif // CUDA_SUPPORTED

using namespace Simple;
using namespace Simple::Display;

using Stars = std::vector<Starfield::Star>;

//--------------------------------------------------------------
Starfield::Starfield(const Config& a_config)
    : m_stars(a_config.numStars)
    , m_cudaDeviceStars(nullptr)
    , m_config(a_config)
{
}

//--------------------------------------------------------------
void Starfield::StartUp()
{
    assert(m_context == nullptr);
    m_context = new Context(m_config.displayContextConfig);

#ifdef CUDA_SUPPORTED
    const size_t starsSize = m_stars.size() * sizeof(Star);
    cudaMalloc(&m_cudaDeviceStars, starsSize);
    cudaMemcpy(m_cudaDeviceStars,
               m_stars.data(),
               starsSize,
               cudaMemcpyHostToDevice);
#endif // CUDA_SUPPORTED

    // Shut down immediately if there is no display buffer.
    if (!m_context->GetBuffer().GetData())
    {
        RequestShutDown();
    }
}

//--------------------------------------------------------------
void Starfield::ShutDown()
{
#ifdef CUDA_SUPPORTED
    cudaFree(m_cudaDeviceStars);
#endif // CUDA_SUPPORTED
    m_cudaDeviceStars = nullptr;

    delete(m_context);
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
            static constexpr float BLACK[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
            static constexpr float WHITE[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
            UpdateStars<float, 4>(BLACK, WHITE, a_fixedTimeSeconds);
        }
        break;
        case Buffer::Format::RGBA_UINT8:
        {
            static constexpr uint8_t BLACK[4] = { 0, 0, 0, UINT8_MAX };
            static constexpr uint8_t WHITE[4] = { UINT8_MAX, UINT8_MAX, UINT8_MAX, UINT8_MAX };
            UpdateStars<uint8_t, 4>(BLACK, WHITE, a_fixedTimeSeconds);
        }
        break;
        case Buffer::Format::RGBA_UINT16:
        {
            static constexpr uint16_t BLACK[4] = { 0, 0, 0, UINT16_MAX };
            static constexpr uint16_t WHITE[4] = { UINT16_MAX, UINT16_MAX, UINT16_MAX, UINT16_MAX };
            UpdateStars<uint16_t, 4>(BLACK, WHITE, a_fixedTimeSeconds);
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
template <typename DataType, uint32_t ChannelsPerPixel>
extern void UpdateStarsCuda(const Starfield::Config& a_config,
                            const Display::Context& a_context,
                            const DataType a_backColor[ChannelsPerPixel],
                            const DataType a_starColor[ChannelsPerPixel],
                            const Stars& a_hostStars,
                            Starfield::Star* a_stars,
                            float a_secondsElapsed);
#endif // CUDA_SUPPORTED

//--------------------------------------------------------------
template <class T = uint32_t>
T RandomFloat(T a_min, T a_max)
{
    static std::random_device s_randomDevice;
    static std::mt19937 s_generator(s_randomDevice());
    std::uniform_real_distribution<T> distribution(a_min, a_max);
    return distribution(s_generator);
}

//--------------------------------------------------------------
template<typename DataType, uint32_t ChannelsPerPixel>
void UpdateStarsHost(const DataType a_backColor[ChannelsPerPixel],
                     const DataType a_starColor[ChannelsPerPixel],
                     const Starfield::Config& a_config,
                     Display::Context& a_context,
                     Stars& a_hostStars,
                     float a_secondsElapsed)
{
    // Cache buffer values.
    const Buffer& buffer = a_context.GetBuffer();
    const uint32_t bufferWidth = buffer.GetWidth();
    const uint32_t bufferHeight = buffer.GetHeight();
    DataType* bufferData = buffer.GetData<DataType>();
    if (!bufferWidth || !bufferHeight || !bufferData)
    {
        return;
    }

    // Clear the display buffer.
    const uint32_t totalChannels = bufferWidth *
                                   bufferHeight *
                                   ChannelsPerPixel;
    assert(ChannelsPerPixel == Buffer::ChannelsPerPixel(buffer.GetFormat()));
    for (uint32_t i = 0; i < totalChannels; i += ChannelsPerPixel)
    {
        for (uint32_t j = 0; j < ChannelsPerPixel; ++j)
        {
            bufferData[i + j] = a_backColor[j];
        }
    }

    // Draw the stars.
    const float sceneWidth = static_cast<float>(bufferWidth);
    const float sceneHeight = static_cast<float>(bufferHeight);
    const float sceneCenterX = sceneWidth * 0.5f;
    const float sceneCenterY = sceneHeight * 0.5f;
    const size_t numStars = a_hostStars.size();
    for (size_t i = 0; i < numStars; ++i)
    {
        Starfield::Star& starInstance = a_hostStars[i];
        if (starInstance.z <= a_config.zNear)
        {
            // The star has passed the near clip plane,
            // position it randomly in the visible range.
            starInstance.x = (RandomFloat(0.0f, 1.0f) * sceneWidth) - sceneCenterX;
            starInstance.y = (RandomFloat(0.0f, 1.0f) * sceneHeight) - sceneCenterY;
            starInstance.z = (RandomFloat(0.0f, 1.0f) * (a_config.zFar - a_config.zNear));
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
                const uint32_t index = ((x * ChannelsPerPixel) +
                                        (y * bufferWidth * ChannelsPerPixel));
                for (uint32_t z = 0; z < ChannelsPerPixel; ++z)
                {
                    bufferData[index + z] = a_starColor[z];
                }
            }
        }
    }
}

//--------------------------------------------------------------
template<typename DataType, uint32_t ChannelsPerPixel>
void Starfield::UpdateStars(const DataType a_backColor[ChannelsPerPixel],
                            const DataType a_starColor[ChannelsPerPixel],
                            float a_secondsElapsed)
{
    const Buffer& buffer = m_context->GetBuffer();
    if (buffer.GetInterop() == Buffer::Interop::HOST)
    {
        UpdateStarsHost<DataType, ChannelsPerPixel>(a_backColor,
                                                    a_starColor,
                                                    m_config,
                                                    *m_context,
                                                    m_stars,
                                                    a_secondsElapsed);
    }
    else if (buffer.GetInterop() == Buffer::Interop::CUDA)
    {
    #ifdef CUDA_SUPPORTED
        UpdateStarsCuda<DataType, ChannelsPerPixel>(m_config,
                                                    *m_context,
                                                    a_backColor,
                                                    a_starColor,
                                                    m_stars,
                                                    m_cudaDeviceStars,
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
