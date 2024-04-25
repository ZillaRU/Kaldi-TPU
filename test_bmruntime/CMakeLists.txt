cmake_minimum_required(VERSION 3.10)
project(test_asr_bmrt)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()
include_directories(/opt/sophon/libsophon-0.5.0/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_INSTALL_PREFIX install)

add_definitions(-DSOC_TARGET)
link_directories(/data/Qwen/lib_soc)

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)

add_executable(testbmrt test_single_bmodel.cc)
target_link_libraries(testbmrt bmrt bmlib)

add_executable(testbmrt3in1 test_3in1bmodel.cc)
target_link_libraries(testbmrt3in1 bmrt bmlib)