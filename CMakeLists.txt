cmake_minimum_required(VERSION 3.10)
project(pipeline LANGUAGES C CXX)

# Set C++ and C standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_EXTENSIONS ON)

# Suppress FLANN_ROOT warning from PCL and set policy
unset(FLANN_ROOT)
cmake_policy(SET CMP0144 NEW)

# Add this hint to point to your custom TBB installation directory.
set(TBB_ROOT "/usr/local")

# --- Find Dependencies ---
# Find all required packages for the project.
find_package(PCL 1.14 REQUIRED COMPONENTS common io filters features segmentation registration visualization)
find_package(VTK REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(GTSAM REQUIRED)
find_library(CPUID_LIBRARY NAMES cpuid libcpuid)

# Check for CPUID for runtime detection
if(CPUID_LIBRARY)
    message(STATUS "Found cpuid library: ${CPUID_LIBRARY}")
else()
    message(STATUS "cpuid library not found; runtime CPU detection disabled")
    add_compile_definitions(NO_CPUID)
endif()

# --- Define Shared Library Sources ---
set(PIPELINE_LIB_SOURCES
    src/lidarcallback.cpp
    src/compcallback.cpp
    src/registercallback.cpp
    extern/ndt_omp/src/estimate_covariance/estimate_covariance.cpp
    extern/ndt_omp/src/multigrid_pclomp/multi_voxel_grid_covariance_omp.cpp
    extern/ndt_omp/src/multigrid_pclomp/multigrid_ndt_omp.cpp
    extern/ndt_omp/src/pclomp/gicp_omp.cpp
    extern/ndt_omp/src/pclomp/ndt_omp.cpp
    extern/ndt_omp/src/pclomp/voxel_grid_covariance_omp.cpp
)

# --- Define Shared Include Directories ---
set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/ndt_omp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/robin-map/include/tsl
    ${PCL_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
)

# --- Create the Shared Library ---
add_library(pipeline_lib ${PIPELINE_LIB_SOURCES})

# --- Configure the Shared Library ---
target_include_directories(pipeline_lib PUBLIC ${INCLUDE_DIRS})

target_link_libraries(pipeline_lib PUBLIC
    ${PCL_LIBRARIES}
    ${VTK_LIBRARIES}
    nlohmann_json::nlohmann_json
    Boost::system
    Boost::thread
    gtsam
    # TBB::tbb and TBB::tbbmalloc are not standard find_package targets.
    # If you have a custom FindTBB.cmake, this is fine. Otherwise, you may
    # need to link to TBB libraries directly using their paths.
)

# --- Set Target-Specific Compiler Flags ---
# High-level optimization for the core library
target_compile_options(pipeline_lib PUBLIC -O3)

# Check for AVX2 and FMA support and apply to the library if available
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
    message(STATUS "AVX2 and FMA are supported by the compiler. Enabling for pipeline_lib.")
    target_compile_options(pipeline_lib PUBLIC -mavx2 -mfma)
else()
    message(WARNING "AVX2 or FMA not supported; using scalar implementation.")
endif()


# --- Build the Executables ---
# # 1. Main pipeline executable
# add_executable(pipeline_ligo_tc run/pipeline_ligo_tc.cpp)
# target_link_libraries(pipeline_ligo_tc PRIVATE pipeline_lib)

# # 2. Compass calibration executable
# add_executable(pipeline_calib_compass run/pipeline_calib_compass.cpp)
# target_link_libraries(pipeline_calib_compass PRIVATE pipeline_lib)

# # 2. Compass calibration executable
# add_executable(pipeline run/pipeline.cpp)
# target_link_libraries(pipeline PRIVATE pipeline_lib)

# 2. Lidar Visualizer executable
add_executable(viz_lidar_udp run/viz_lidar_udp.cpp)
target_link_libraries(viz_lidar_udp PRIVATE pipeline_lib)

# 2. Lidar Visualizer executable
add_executable(pipeline_ins_map_distribution run/pipeline_ins_map_distribution.cpp)
target_link_libraries(pipeline_ins_map_distribution PRIVATE pipeline_lib)
# The optimization and AVX2 flags are inherited from pipeline_lib, so they're not needed here.

# --- Final Checks ---
if(NOT VTK_FOUND)
    message(FATAL_ERROR "VTK not found. Please install VTK.")
endif()

