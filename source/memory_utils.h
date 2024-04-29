//--------------------------------------------------------------
// Copyright (c) David Bosnich <david.bosnich.public@gmail.com>
//
// This code is licensed under the MIT License, a copy of which
// can be found in the license.txt file included at the root of
// this distribution, or at https://opensource.org/licenses/MIT
//--------------------------------------------------------------

#pragma once

#ifdef _WIN32
#   define _CRTDBG_MAP_ALLOC
#   include <stdlib.h>
#   include <crtdbg.h>

#   ifdef _DEBUG
#       define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#       define new DEBUG_NEW
#   endif // _DEBUG

#   define NOMINMAX
#   include <windows.h>
#   include <psapi.h>
#endif

#include <stdint.h>
#include <stdio.h>

//--------------------------------------------------------------
namespace Simple
{

//--------------------------------------------------------------
namespace MemoryUtils
{

void PrintMemoryUsage(const char* a_additionalInfo);
void PrintMemoryLeaks(const char* a_additionalInfo);
struct LeakDetector
{
    ~LeakDetector();
};

//--------------------------------------------------------------
inline uint32_t GetMemoryUsageBytes()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS processInfo;
    const BOOL result = GetProcessMemoryInfo(GetCurrentProcess(),
                                             &processInfo,
                                             sizeof(processInfo));
    return result ? (uint32_t)processInfo.WorkingSetSize : 0;
#else
    printf("MemoryUtils::GetMemoryUsageBytes not implemented\n");
    return 0;
#endif
}

//--------------------------------------------------------------
inline void PrintMemoryUsage(const char* a_additionalInfo)
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS processInfo;
    if (::GetProcessMemoryInfo(::GetCurrentProcess(),
                               &processInfo,
                               sizeof(processInfo)))
    {
        printf("Current Memory Usage (in bytes): %zu (%s)\n",
               processInfo.WorkingSetSize, a_additionalInfo);
    }
    else
    {
        printf("Could not get current process memory info.\n");
    }
#else
    printf("MemoryUtils::PrintMemoryUsage not implemented\n");
    (void)a_additionalInfo;
#endif
}

//--------------------------------------------------------------
inline void PrintMemoryLeaks(const char* a_additionalInfo)
{
#ifndef _DEBUG
    (void)a_additionalInfo;
#elif defined(_WIN32)
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDOUT);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDOUT);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDOUT);
    const char* output = _CrtDumpMemoryLeaks() == TRUE ?
                         "Memory leak(s) detected (%s)\n" :
                         "No memory leaks detected (%s)\n";
    printf(output, a_additionalInfo);
#else
    printf("MemoryUtils::PrintMemoryLeaks not implemented\n");
    (void)a_additionalInfo;
#endif
}

//--------------------------------------------------------------
inline LeakDetector::~LeakDetector()
{
    PrintMemoryLeaks("LeakDetector");
}

} // namespace MemoryUtils
} // namespace Simple
