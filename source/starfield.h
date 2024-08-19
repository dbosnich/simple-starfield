//--------------------------------------------------------------
// Copyright (c) David Bosnich <david.bosnich.public@gmail.com>
//
// This code is licensed under the MIT License, a copy of which
// can be found in the license.txt file included at the root of
// this distribution, or at https://opensource.org/licenses/MIT
//--------------------------------------------------------------

#pragma once

#include <simple/application/application.h>
#include <simple/display/context.h>

#include <vector>

//--------------------------------------------------------------
namespace Simple
{

//--------------------------------------------------------------
class Starfield : public Application
{
public:
    //----------------------------------------------------------
    struct Config
    {
        float zNear = 0.0f;
        float zFar = 800.0f;
        float focalLength = 200.0f;
        float starWidth = 3.0f;
        float starHeight = 3.0f;
        float starSpeed = 60.0f;
        uint32_t numStars = 5000;
        Display::Context::Config displayContextConfig = {};
    };

    //----------------------------------------------------------
    struct Star
    {
        float x = 0.0f, y = 0.0f, z = 0.0f;
        float r = 1.0f, g = 1.0f, b = 1.0f;
    };

    Starfield(const Config& a_starfieldConfig);
    virtual ~Starfield() = default;

    Starfield(const Starfield&) = delete;
    Starfield& operator= (const Starfield&) = delete;

protected:
    void StartUp() override;
    void ShutDown() override;

    void UpdateStart(float a_deltaTimeSeconds) override;
    void UpdateFixed(float a_fixedTimeSeconds) override;
    void UpdateEnded(float a_deltaTimeSeconds) override;

    template<typename DataType>
    void UpdateStars(float a_timeSeconds);

    void OnFrameComplete(const FrameStats& a_stats) override;

private:
    Display::Context* m_context = nullptr;
    std::vector<Star> m_stars;
    Star* m_cudaDeviceStars;
    const Config m_config;
};

} // namespace Simple
