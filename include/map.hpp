#pragma once
#include <vector>
#include <robin_map.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <dataframe.hpp>

struct Voxel {
    int32_t x = 0; // voxel index in x-direction
    int32_t y = 0; // voxel index in y-direction
    int32_t z = 0; // voxel index in z-direction

    // %            ... voxel constructor
    Voxel(int x, int y, int z) : x(static_cast<int32_t>(x)), y(static_cast<int32_t>(y)), z(static_cast<int32_t>(z)) {}

    // %            ... equality operation for comparing 2 voxels
    bool operator==(const Voxel& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    // %            ... less than operation for comparing 2 voxels
    bool operator<(const Voxel& other) const {
        return x < other.x || (x == other.x && y < other.y) || (x == other.x && y == other.y && z < other.z);
    }

    // %            ... convert a point into voxel coordinates
    static Voxel getKey(const Eigen::Vector3f& point, double voxel_size) {
        return {
            static_cast<int32_t>(point.x() / voxel_size),
            static_cast<int32_t>(point.y() / voxel_size),
            static_cast<int32_t>(point.z() / voxel_size)
        };
    }
};

struct VoxelHash {
    // %            ... hash combining pattern
    static size_t hash(const Voxel& voxel) {
        size_t seed = 0;
        seed ^= std::hash<int32_t>()(voxel.x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int32_t>()(voxel.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int32_t>()(voxel.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

    // %            ... hash equal comparison
    static bool equal(const Voxel& v1, const Voxel& v2) {
        return v1 == v2;
    }

    // %            ... operation for calling the static hash function
    size_t operator()(const Voxel& voxel) const {
        return hash(voxel);
    }
};

struct KeyframeHash {
    size_t operator()(uint64_t k) const {
        // 1. Add a large golden-ratio-based constant to start mixing.
        k += 0x9e3779b97f4a7c15;
        
        // 2. XOR-shift and multiply with large prime constants to avalanche bits.
        k = (k ^ (k >> 30)) * 0xbf58476d1ce4e5b9;
        k = (k ^ (k >> 27)) * 0x94d049bb133111eb;
        
        // 3. Final XOR-shift to ensure all bits are well-mixed.
        k = k ^ (k >> 31);
        
        return static_cast<size_t>(k);
    }
};

struct KeyframeInfo {
    uint64_t id;
    double timestamp;
};
struct KeypointInfo{
    pcl::PointCloud<pcl::PointXYZI>::Ptr points;
    double timestamp;
};
struct Keypose {
    Eigen::Matrix4d pose; // Tb2m: body-to-map transformation in NED frame
    double timestamp;
};
struct KeyFrameStats {
    Eigen::Vector3d rlla = Eigen::Vector3d::Zero(); 
    uint64_t frame_id = 0;
    double timestamp = 0.0;
    size_t num_points = 0;
    long alignment_time_ms = 0;
    int ndt_iterations = 0;
    
    Eigen::Vector<double, 6> ins_std_dev = Eigen::Vector<double, 6>::Zero();
    Eigen::Vector<double, 6> ins_scaled_std_dev = Eigen::Vector<double, 6>::Zero();
    Eigen::Vector<double, 6> lidar_std_dev = Eigen::Vector<double, 6>::Zero();
    Eigen::Vector<double, 6> gtsam_std_dev = Eigen::Vector<double, 6>::Zero();
    
    Eigen::Matrix4d ins_pose = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d gtsam_pose = Eigen::Matrix4d::Identity();
    
    double pose_rmse = 0.0; // RMSE between INS translation and GTSAM translation
};


using VoxelHashMap = tsl::robin_map<Voxel, std::vector<KeyframeInfo>, VoxelHash>;
using PointsHashMap = tsl::robin_map<uint64_t, KeypointInfo, KeyframeHash>;
using PoseHashMap = tsl::robin_map<uint64_t, Keypose, KeyframeHash>;
using StatsHashMap = tsl::robin_map<uint64_t, KeyFrameStats, KeyframeHash>;
