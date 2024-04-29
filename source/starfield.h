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

//--------------------------------------------------------------
namespace Simple
{

//--------------------------------------------------------------
class Starfield : public Application
{
public:
    Starfield() = default;
    virtual ~Starfield() = default;

    Starfield(const Starfield&) = delete;
    Starfield& operator= (const Starfield&) = delete;

protected:
    void StartUp() override;
    void ShutDown() override;

    void UpdateStart(float a_deltaTimeSeconds) override;
    void UpdateFixed(float a_fixedTimeSeconds) override;
    void UpdateEnded(float a_deltaTimeSeconds) override;

    template<typename DataType, uint32_t ChannelsPerPixel>
    void UpdateStars(const DataType a_backColor[ChannelsPerPixel],
                     const DataType a_starColor[ChannelsPerPixel],
                     float a_timeSeconds);

    void OnFrameComplete(const FrameStats& a_stats) override;

private:
    Display::Context* m_context = nullptr;
};

} // namespace Simple
