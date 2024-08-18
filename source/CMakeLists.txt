##--------------------------------------------------------------
## Copyright (c) David Bosnich <david.bosnich.public@gmail.com>
##
## This code is licensed under the MIT License, a copy of which
## can be found in the license.txt file included at the root of
## this distribution, or at https://opensource.org/licenses/MIT
##--------------------------------------------------------------

# Set the source directory.
set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/source")

# Gather all source files.
file(GLOB_RECURSE source_files ${SOURCE_DIR}/*.h
                               ${SOURCE_DIR}/*.cpp
                               ${SOURCE_DIR}/*.cu)

# Group all files for the IDE.
source_group(TREE "${SOURCE_DIR}"
             PREFIX "source"
             FILES ${source_files})

# Fetch the SimpleApplication dependency.
Include(FetchContent)
FetchContent_Declare(
    simple_application
    GIT_REPOSITORY https://github.com/dbosnich/simple-application.git
    GIT_TAG main)
FetchContent_MakeAvailable(simple_application)

# Group SimpleApplication files for the IDE.
get_target_property(application_header_files simple_application INTERFACE_SOURCES)
get_target_property(application_source_dir simple_application SOURCE_DIR)
source_group(TREE "${application_source_dir}/include"
             PREFIX "include"
             FILES ${application_header_files})

# Fetch the SimpleDisplay dependency.
FetchContent_Declare(
    simple_display
    GIT_REPOSITORY https://github.com/dbosnich/simple-display.git
    GIT_TAG main)
FetchContent_MakeAvailable(simple_display)

# Group SimpleDisplay files for the IDE.
get_target_property(display_header_files simple_display INTERFACE_SOURCES)
get_target_property(display_source_dir simple_display SOURCE_DIR)
cmake_path(GET display_source_dir PARENT_PATH display_root_dir)
source_group(TREE "${display_root_dir}/include"
             PREFIX "include"
             FILES ${display_header_files})

# Define the application.
set(APP_TARGET "${PROJECT_NAME}")
add_executable(${APP_TARGET} ${source_files})
target_link_libraries(${APP_TARGET} simple_application
                                    simple_display)
target_compile_features(${APP_TARGET} PUBLIC cxx_std_14)
target_include_directories(${APP_TARGET} PRIVATE ${SOURCE_DIR})
target_compile_options(${APP_TARGET} PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:
    $<$<CXX_COMPILER_ID:MSVC>: /GR- /W4 /WX>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fno-rtti -Wall -Werror -Wextra>
  >
)

# Add CUDA dependencies.
find_package(CUDAToolkit)
if (${CUDAToolkit_FOUND})
    enable_language(CUDA)
    target_compile_definitions(${APP_TARGET} PRIVATE CUDA_SUPPORTED)
else()
    # Ensure CUDA files are displayed but not built.
    file(GLOB_RECURSE cuda_files *.cu)
    set_source_files_properties(${cuda_files} PROPERTIES HEADER_FILE_ONLY TRUE)
endif()
