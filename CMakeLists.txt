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

# --- Define Shared Library Sources ---
# These are all the .cpp files that contain the core logic of your application,
# excluding any files with a main() function.
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
# These directories are needed by the library and the executables.
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
# This command compiles all the source files above into a single library
# named "pipeline_lib".
add_library(pipeline_lib ${PIPELINE_LIB_SOURCES})

# --- Configure the Shared Library ---
# Set the necessary include paths and link the dependencies for the library.
# Using PUBLIC ensures that any executable linking against pipeline_lib
# will automatically inherit these properties.
target_include_directories(pipeline_lib PUBLIC ${INCLUDE_DIRS})

target_link_libraries(pipeline_lib PUBLIC
    ${PCL_LIBRARIES}
    ${VTK_LIBRARIES}
    nlohmann_json::nlohmann_json
    Boost::system
    Boost::thread
    gtsam
    TBB::tbb
    TBB::tbbmalloc
)

# --- Build the Executables ---
# Now, define each program. They are much simpler because all the heavy
# lifting is done by the library.

# 1. Main pipeline executable
add_executable(pipeline_ligo_tc run/pipeline_ligo_tc.cpp)
target_link_libraries(pipeline_ligo_tc PRIVATE pipeline_lib)
target_compile_options(pipeline_ligo_tc PRIVATE -O3)

# 2. Compass calibration executable
add_executable(pipeline_calib_compass run/pipeline_calib_compass.cpp)
target_link_libraries(pipeline_calib_compass PRIVATE pipeline_lib)
target_compile_options(pipeline_calib_compass PRIVATE -O3)


# --- Final Checks ---
if(NOT VTK_FOUND)
    message(FATAL_ERROR "VTK not found. Please install VTK.")
endif()
