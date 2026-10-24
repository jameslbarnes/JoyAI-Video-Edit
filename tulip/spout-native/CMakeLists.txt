cmake_minimum_required(VERSION 3.20)
project(joyai_spout LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT WIN32 OR NOT EXISTS "${SPOUT_ROOT}/SPOUTSDK/SpoutDirectX/SpoutDX/SpoutDX.cpp")
  message(FATAL_ERROR "Windows and -DSPOUT_ROOT=<Spout2 2.007.018 checkout> required")
endif()
if(NOT EXISTS "${CUDA_ROOT}/include/cuda.h")
  message(FATAL_ERROR "-DCUDA_ROOT=<CUDA development headers and lib/x64> required")
endif()
set(GL "${SPOUT_ROOT}/SPOUTSDK/SpoutGL")
set(SPOUT_SOURCES
  "${SPOUT_ROOT}/SPOUTSDK/SpoutDirectX/SpoutDX/SpoutDX.cpp"
  "${GL}/SpoutCopy.cpp" "${GL}/SpoutDirectX.cpp" "${GL}/SpoutFrameCount.cpp"
  "${GL}/SpoutSenderNames.cpp" "${GL}/SpoutSharedMemory.cpp" "${GL}/SpoutUtils.cpp")
add_library(joyai_spout SHARED sender.cpp ${SPOUT_SOURCES})
add_library(joyai_spout_input SHARED receiver.cpp ${SPOUT_SOURCES})
add_executable(test_spout_present_trace test_present_trace.cpp)
add_executable(test_spout_frame_trace test_frame_trace.cpp)
foreach(target joyai_spout joyai_spout_input)
target_include_directories(${target} PRIVATE "${CUDA_ROOT}/include"
  "${SPOUT_ROOT}/SPOUTSDK/SpoutDirectX/SpoutDX" "${GL}")
target_compile_definitions(${target} PRIVATE NOMINMAX SPOUT_BUILD_STATIC)
target_link_libraries(${target} PRIVATE "${CUDA_ROOT}/lib/x64/cuda.lib"
  d3d11 dxgi dxguid d3d9 shlwapi winmm avrt version)
endforeach()
