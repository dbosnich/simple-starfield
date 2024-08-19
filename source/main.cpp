//--------------------------------------------------------------
// Copyright (c) David Bosnich <david.bosnich.public@gmail.com>
//
// This code is licensed under the MIT License, a copy of which
// can be found in the license.txt file included at the root of
// this distribution, or at https://opensource.org/licenses/MIT
//--------------------------------------------------------------

#include <memory_utils.h>
#include <starfield.h>
#include <string.h>

static Simple::MemoryUtils::LeakDetector s_leakDetector;

using namespace Simple::Display;
using namespace std;

//--------------------------------------------------------------
void SetWindowTitle(Context::Config& a_contextConfig)
{
    string graphicsAPI;
    switch (a_contextConfig.graphicsAPI)
    {
        case Context::GraphicsAPI::NATIVE: graphicsAPI = "GraphicsAPI::NATIVE"; break;
        case Context::GraphicsAPI::OPENGL: graphicsAPI = "GraphicsAPI::OPENGL"; break;
        case Context::GraphicsAPI::VULKAN: graphicsAPI = "GraphicsAPI::VULKAN"; break;
        default: graphicsAPI = "GraphicsAPI::NONE"; break;
    }

    string format;
    switch (a_contextConfig.bufferConfig.format)
    {
        case Buffer::Format::RGBA_FLOAT: format = "Format::RGBA_FLOAT"; break;
        case Buffer::Format::RGBA_UINT8: format = "Format::RGBA_UINT8"; break;
        case Buffer::Format::RGBA_UINT16: format = "Format::RGBA_UINT16"; break;
        default: format = "Format::NONE"; break;
    }

    string interop;
    switch (a_contextConfig.bufferConfig.interop)
    {
        case Buffer::Interop::HOST: interop = "Interop::HOST"; break;
        case Buffer::Interop::CUDA: interop = "Interop::CUDA"; break;
        default: interop = "Interop::NONE"; break;
    }

    const string title = graphicsAPI + " " + format + " " + interop;
    a_contextConfig.windowConfig.titleUTF8 = title;
}

//--------------------------------------------------------------
int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    Simple::Starfield::Config config;
    config.displayContextConfig.graphicsAPI = Simple::Display::Context::GraphicsAPI::NATIVE;
    config.displayContextConfig.bufferConfig.format = Simple::Display::Buffer::Format::RGBA_UINT8;
    config.displayContextConfig.bufferConfig.interop = Simple::Display::Buffer::Interop::CUDA;
    SetWindowTitle(config.displayContextConfig);
    Simple::Starfield starfield1(config);

    config.displayContextConfig.graphicsAPI = Simple::Display::Context::GraphicsAPI::NATIVE;
    config.displayContextConfig.bufferConfig.format = Simple::Display::Buffer::Format::RGBA_UINT8;
    config.displayContextConfig.bufferConfig.interop = Simple::Display::Buffer::Interop::HOST;
    SetWindowTitle(config.displayContextConfig);
    Simple::Starfield starfield2(config);

    config.displayContextConfig.graphicsAPI = Simple::Display::Context::GraphicsAPI::OPENGL;
    config.displayContextConfig.bufferConfig.format = Simple::Display::Buffer::Format::RGBA_UINT8;
    config.displayContextConfig.bufferConfig.interop = Simple::Display::Buffer::Interop::CUDA;
    SetWindowTitle(config.displayContextConfig);
    Simple::Starfield starfield3(config);

    config.displayContextConfig.graphicsAPI = Simple::Display::Context::GraphicsAPI::OPENGL;
    config.displayContextConfig.bufferConfig.format = Simple::Display::Buffer::Format::RGBA_UINT8;
    config.displayContextConfig.bufferConfig.interop = Simple::Display::Buffer::Interop::HOST;
    SetWindowTitle(config.displayContextConfig);
    Simple::Starfield starfield4(config);

    config.displayContextConfig.graphicsAPI = Simple::Display::Context::GraphicsAPI::VULKAN;
    config.displayContextConfig.bufferConfig.format = Simple::Display::Buffer::Format::RGBA_UINT8;
    config.displayContextConfig.bufferConfig.interop = Simple::Display::Buffer::Interop::CUDA;
    SetWindowTitle(config.displayContextConfig);
    Simple::Starfield starfield5(config);

    config.displayContextConfig.graphicsAPI = Simple::Display::Context::GraphicsAPI::VULKAN;
    config.displayContextConfig.bufferConfig.format = Simple::Display::Buffer::Format::RGBA_UINT8;
    config.displayContextConfig.bufferConfig.interop = Simple::Display::Buffer::Interop::HOST;
    SetWindowTitle(config.displayContextConfig);
    Simple::Starfield starfield6(config);

    // MacOS display contexts can only run on the main thread.
#ifdef __APPLE__
    starfield1.Run();
    starfield2.Run();
    starfield3.Run();
    starfield4.Run();
    starfield5.Run();
    starfield6.Run();
#else
    std::thread runThread1 = starfield1.RunInThread();
    std::thread runThread2 = starfield2.RunInThread();
    std::thread runThread3 = starfield3.RunInThread();
    std::thread runThread4 = starfield4.RunInThread();
    std::thread runThread5 = starfield5.RunInThread();
    std::thread runThread6 = starfield6.RunInThread();

    runThread1.join();
    runThread2.join();
    runThread3.join();
    runThread4.join();
    runThread5.join();
    runThread6.join();
#endif // __APPLE__
}
