cmake_minimum_required(VERSION 3.10)
project(pipeline LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON) # Enable GNU extensions for C++
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_EXTENSIONS ON)   # Enable GNU extensions for C

# Find dependencies
find_package(PCL 1.14 REQUIRED COMPONENTS common io filters features segmentation registration visualization)
find_package(VTK REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)

# Define source files
set(SOURCES
    run/pipeline.cpp
    src/lidarcallback.cpp
    src/compcallback.cpp
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
    ${PCL_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

# Create executable
add_executable(pipeline ${SOURCES})

# Set include directories for the target
target_include_directories(pipeline PRIVATE ${INCLUDE_DIRS})

# Link libraries
target_link_libraries(pipeline PRIVATE
    ${PCL_LIBRARIES}
    ${VTK_LIBRARIES}
    nlohmann_json::nlohmann_json
    Boost::system
    Boost::thread
)

# Add compile options for optimization
target_compile_options(pipeline PRIVATE -O3)

# Ensure VTK is properly configured
if(VTK_FOUND)
    include(${VTK_USE_FILE})
endif()