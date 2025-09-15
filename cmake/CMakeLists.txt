cmake_minimum_required(VERSION 3.16)
project(sam LANGUAGES C CXX)

# ----------------------------------------------------------------------------
# Compiler Configuration
# ----------------------------------------------------------------------------
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) # Enable GNU extensions for C++
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_EXTENSIONS ON)   # Enable GNU extensions for C

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")

# ----------------------------------------------------------------------------
# CMake Policies
# ----------------------------------------------------------------------------
if(POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW) # Address Open3D deprecation warning
endif()
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW) # Address FLANN_ROOT warning
endif()

# ----------------------------------------------------------------------------
# Find Dependencies
# ----------------------------------------------------------------------------

set(Open3D_DIR "/usr/local/lib/cmake/Open3D/") # Path to Open3D CMake config
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Open3D REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(nlohmann_json REQUIRED)
find_package(OpenMP)
find_package(PCL REQUIRED COMPONENTS common io filters features segmentation registration)

# Find custom TBB library
find_library(TBB_LIBRARY NAMES tbb libtbb HINTS /usr/local/lib/)
if(TBB_LIBRARY)
    message(STATUS "Found custom TBB library: ${TBB_LIBRARY}")
else()
    message(WARNING "Custom TBB library not found in /usr/local/lib/; linking may fail")
endif()

# Find cpuid library
find_library(CPUID_LIBRARY NAMES cpuid libcpuid)
if(CPUID_LIBRARY)
    message(STATUS "Found cpuid library: ${CPUID_LIBRARY}")
else()
    message(STATUS "cpuid library not found; runtime CPU detection disabled")
    add_compile_definitions(NO_CPUID)
endif()

# ----------------------------------------------------------------------------
# Optional Features and Compiler Checks
# ----------------------------------------------------------------------------
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found; enabling parallelization")
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
    message(STATUS "AVX2 and FMA supported; enabling vectorized instructions")
    add_compile_options(-mavx2 -mfma)
else()
    message(WARNING "AVX2 or FMA not supported; using scalar implementation")
endif()

# ----------------------------------------------------------------------------
# RPATH Configuration for Custom TBB
# ----------------------------------------------------------------------------
set(CMAKE_BUILD_RPATH "/usr/local/lib/")
set(CMAKE_INSTALL_RPATH "/usr/local/lib/")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# ----------------------------------------------------------------------------
# Project Structure
# ----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
endif()

# Core library for SLAM components
add_library(core STATIC
    src/compcallback.cpp
    src/lidarcallback.cpp
    src/pipeline.cpp
    extern/pclomp/src/pclomp/gicp_omp.cpp
    extern/pclomp/src/pclomp/ndt_omp.cpp
    extern/pclomp/src/pclomp/voxel_grid_covariance_omp.cpp
)

target_include_directories(core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/robin-map/include/tsl
    ${CMAKE_SOURCE_DIR}/extern/Sophus
    ${CMAKE_SOURCE_DIR}/extern/pclomp/src
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(core PUBLIC
    Boost::system
    Open3D::Open3D
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    ${PCL_LIBRARIES}
    ${TBB_LIBRARY}
)

# Main executable
add_executable(sam src/main.cpp)

target_include_directories(sam PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/robin-map/include/tsl
    ${CMAKE_SOURCE_DIR}/extern/Sophus
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(sam PRIVATE core)

# Add PCL definitions
add_definitions(${PCL_DEFINITIONS})