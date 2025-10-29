cmake_minimum_required(VERSION 3.10)
project(SvnNdtTest)

# Set policy to suppress the FLANN_ROOT warning
cmake_policy(SET CMP0144 NEW)

# Add this hint to point to your custom TBB installation directory.
set(TBB_ROOT "/usr/local")

find_package(PCL 1.8 REQUIRED)
find_package(GTSAM 4.0 REQUIRED)
find_package(GTest REQUIRED)
find_package(TBB REQUIRED) # Your svn_ndt_impl.hpp uses TBB

# Add the directory containing your svn_ndt.h and voxel_grid_covariance.h
# e.g., include_directories(path/to/your/include)
# Add "include" to this line
include_directories(
    include 
    ${CMAKE_CURRENT_SOURCE_DIR}/../robin-map/include # Path to robin-map
    ${CMAKE_CURRENT_SOURCE_DIR}/../ndt_omp/include
    ${GTSAM_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS} 
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(run_svn_ndt_test 
test/test_svn_ndt.cpp
../ndt_omp/src/pclomp/ndt_omp.cpp             # PCLOMP NDT implementation (ADJUST PATH IF NEEDED)
../ndt_omp/src/pclomp/voxel_grid_covariance_omp.cpp # PCLOMP VoxelGrid implementation (ADJUST PATH IF NEEDED)
)

target_link_libraries(run_svn_ndt_test
    GTest::gtest_main
    ${PCL_LIBRARIES}
    gtsam # <--- THE FIX IS HERE
    TBB::tbb # Link TBB
)

# Enable C++17 (gtsam, tsl::robin_map, etc. often need it)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  gtest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main
)
FetchContent_MakeAvailable(gtest)