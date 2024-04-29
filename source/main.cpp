//--------------------------------------------------------------
// Copyright (c) David Bosnich <david.bosnich.public@gmail.com>
//
// This code is licensed under the MIT License, a copy of which
// can be found in the license.txt file included at the root of
// this distribution, or at https://opensource.org/licenses/MIT
//--------------------------------------------------------------

#include <memory_utils.h>
#include <starfield.h>

static Simple::MemoryUtils::LeakDetector s_leakDetector;

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    Simple::Starfield starfield;
    starfield.SetCappedFPS(false);
    starfield.Run();
}
