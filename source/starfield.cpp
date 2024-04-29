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

using namespace Simple;
using namespace Simple::Display;

namespace
{
    static constexpr uint32_t WINDOW_WIDTH = 800;
    static constexpr uint32_t WINDOW_HEIGHT = 600;

    static constexpr uint32_t BUFFER_WIDTH = 1920;
    static constexpr uint32_t BUFFER_HEIGHT = 1080;

    static constexpr float Z_NEAR = 0.0f;
    static constexpr float Z_FAR = 800.0f;
    static constexpr float FOCAL_LENGTH = 200.0f; // Distance from camera to viewplane.

    static constexpr float STAR_WIDTH = 3.0f;
    static constexpr float STAR_HEIGHT = 3.0f;

    static constexpr float STAR_SPEED = 60.0f;
    static constexpr uint32_t NUM_STARS = 5000;

    struct Vector3D { float x, y, z = 0.0f; };
    static Vector3D STAR_INSTANCES[NUM_STARS] = {};

    struct ColorFloat { float r, g, b, a = 0.0f; };
    struct ColorUint8 { uint8_t r, g, b, a = 0; };
    struct ColorUint16 { uint16_t r, g, b, a = 0; };
}

//--------------------------------------------------------------
void Starfield::StartUp()
{
    Context::Config contextConfig;
    contextConfig.bufferConfig.width = BUFFER_WIDTH;
    contextConfig.bufferConfig.height = BUFFER_HEIGHT;
    contextConfig.windowConfig.initialWidth = WINDOW_WIDTH;
    contextConfig.windowConfig.initialHeight = WINDOW_HEIGHT;
    contextConfig.windowConfig.titleUTF8 = "Simple Starfield";
    contextConfig.graphicsAPI = Context::GraphicsAPI::NATIVE;

    assert(m_context == nullptr);
    m_context = new Context(contextConfig);
}

//--------------------------------------------------------------
void Starfield::ShutDown()
{
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
            static constexpr uint8_t MAX = std::numeric_limits<uint8_t>::max();
            static constexpr uint8_t BLACK[4] = { 0, 0, 0, MAX };
            static constexpr uint8_t WHITE[4] = { MAX, MAX, MAX, MAX };
            UpdateStars<uint8_t, 4>(BLACK, WHITE, a_fixedTimeSeconds);
        }
        break;
        case Buffer::Format::RGBA_UINT16:
        {
            static constexpr uint16_t MAX = std::numeric_limits<uint16_t>::max();
            static constexpr uint16_t BLACK[4] = { 0, 0, 0, MAX };
            static constexpr uint16_t WHITE[4] = { MAX, MAX, MAX, MAX };
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
void Starfield::UpdateStars(const DataType a_backColor[ChannelsPerPixel],
                            const DataType a_starColor[ChannelsPerPixel],
                            float a_timeSeconds)
{
    // Cache buffer values.
    const Buffer& buffer = m_context->GetBuffer();
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
    for (uint32_t i = 0; i < NUM_STARS; ++i)
    {
        Vector3D& starInstance = STAR_INSTANCES[i];
        if (starInstance.z <= Z_NEAR)
        {
            // The star has passed the near clip plane,
            // position it randomly in the visible range.
            starInstance.x = (RandomFloat(0.0f, 1.0f) * sceneWidth) - sceneCenterX;
            starInstance.y = (RandomFloat(0.0f, 1.0f) * sceneHeight) - sceneCenterY;
            starInstance.z = (RandomFloat(0.0f, 1.0f) * (Z_FAR - Z_NEAR));
        }
        else
        {
            // The star is still visible,
            // move it towards the camera.
            starInstance.z -= STAR_SPEED * a_timeSeconds;
        }

        // Project the star onto the screen.
        const float posX = ((starInstance.x * FOCAL_LENGTH) / starInstance.z) + sceneCenterX;
        const float posY = ((starInstance.y * FOCAL_LENGTH) / starInstance.z) + sceneCenterY;

        // Scale the star such that it gets bigger as it moves towards the screen.
        const float scale = (1.0f - (starInstance.z / Z_FAR));
        const float width = (STAR_WIDTH * scale);
        const float height = (STAR_HEIGHT * scale);

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
