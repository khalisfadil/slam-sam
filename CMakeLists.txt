cmake_minimum_required(VERSION 3.10)
project(pipeline LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) # Enable GNU extensions for C++
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_EXTENSIONS ON) # Enable GNU extensions for C

# Suppress FLANN_ROOT warning from PCL
unset(FLANN_ROOT)
cmake_policy(SET CMP0144 NEW)

# --- NEW LINE ---
# Add this hint to point to your custom TBB installation directory.
set(TBB_ROOT "/usr/local")

# Find dependencies
# When find_package(GTSAM) runs, it will now find TBB in the location specified by TBB_ROOT.
find_package(PCL 1.14 REQUIRED COMPONENTS common io filters features segmentation registration visualization)
find_package(VTK REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(GTSAM REQUIRED)

# Define source files
set(SOURCES
    run/pipeline.cpp
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

# Define include directories
set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/ndt_omp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/robin-map/include/tsl
    ${PCL_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
)

# The manual RPATH settings are no longer needed
# set(CMAKE_INSTALL_RPATH "/usr/local/lib")
# set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

# Create executable
add_executable(pipeline ${SOURCES})

# Set include directories for the target
target_include_directories(pipeline PRIVATE ${INCLUDE_DIRS})

# Link libraries.
target_link_libraries(pipeline PRIVATE
    ${PCL_LIBRARIES}
    ${VTK_LIBRARIES}
    nlohmann_json::nlohmann_json
    Boost::system
    Boost::thread
    gtsam
    TBB::tbb
    TBB::tbbmalloc
)

# Add compile options for optimization
target_compile_options(pipeline PRIVATE -O3)

# Check VTK availability
if(NOT VTK_FOUND)
    message(FATAL_ERROR "VTK not found. Please install VTK.")
endif()