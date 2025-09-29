#pragma once

#include <iostream>
#include <thread>
#include <chrono>
#include <cstdint>

#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>

#include <Eigen/Dense>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <pcl/visualization/pcl_visualizer.h> 
#include <pcl/filters/voxel_grid.h>          
#include <pcl/common/transforms.h>           
#include <pcl/point_types.h>  
#include <pcl/filters/passthrough.h>

#include <lidarcallback.hpp>
#include <compcallback.hpp>
#include <registercallback.hpp>
#include <udpsocket.hpp>
#include <dataframe.hpp>
#include <map.hpp>

using gtsam::Symbol;

template<typename T>
class FrameQueue {
    public:
        void push(std::unique_ptr<T> frame) {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(frame));
            cv_.notify_one();
        }
        std::unique_ptr<T> pop() {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !queue_.empty() || stopped_; });
            if (queue_.empty() && stopped_) return nullptr;
            std::unique_ptr<T> frame = std::move(queue_.front());
            queue_.pop();
            return frame;
        }
        void stop() {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
            cv_.notify_all();
        }
        size_t size() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }
    private:
        std::queue<std::unique_ptr<T>> queue_;
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        bool stopped_ = false;
};

// void manualTransformPointCloud_RowBased(
//         const pcl::PointCloud<pcl::PointXYZI>& cloud_in,
//         pcl::PointCloud<pcl::PointXYZI>& cloud_out,
//         const Eigen::Matrix4f& transform)
//     {
//         cloud_out.clear();
//         cloud_out.resize(cloud_in.size());

//         // Extract the 3x3 rotation matrix (R).
//         Eigen::Matrix3f rotation_matrix = transform.block<3, 3>(0, 0);
//         // Extract the translation vector and represent it as a 1x3 row vector.
//         Eigen::RowVector3f translation_row_vector = transform.block<3, 1>(0, 3).transpose();

//         for (size_t i = 0; i < cloud_in.size(); ++i) {
//             const auto& p_in = cloud_in.points[i];

//             // Represent the input point as a 1x3 Eigen Row Vector.
//             Eigen::RowVector3f p_in_row_vec(p_in.x, p_in.y, p_in.z);

//             // --- THE CORE ROW-BASED TRANSFORMATION ---
//             // Mimics MATLAB: p' = p * R' + t
//             Eigen::RowVector3f p_out_row_vec = p_in_row_vec * rotation_matrix.transpose() + translation_row_vector;

//             auto& p_out = cloud_out.points[i];
//             p_out.y = p_out_row_vec.x();
//             p_out.x = p_out_row_vec.y();
//             p_out.z = p_out_row_vec.z();
//             p_out.intensity = p_in.intensity;
//         }
//     }

void writeStatsToFile(const StatsHashMap& stats, const std::string& filename) {
    if (stats.empty()) {
        std::cout << "No stats to write." << std::endl;
        return;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing stats: " << filename << std::endl;
        return;
    }

    // Write header with new rlla columns
    file << "frame_id,timestamp,num_points,align_time_ms,ndt_iter,"
         // Reference LLA
         << "rlla_lat,rlla_lon,rlla_alt,"
         // Unscaled INS
         << "ins_unscaled_std_x,ins_unscaled_std_y,ins_unscaled_std_z,ins_unscaled_std_roll,ins_unscaled_std_pitch,ins_unscaled_std_yaw,"
         // Scaled INS
         << "ins_scaled_std_x,ins_scaled_std_y,ins_scaled_std_z,ins_scaled_std_roll,ins_scaled_std_pitch,ins_scaled_std_yaw,"
         // Lidar
         << "lidar_std_x,lidar_std_y,lidar_std_z,lidar_std_roll,lidar_std_pitch,lidar_std_yaw,"
         // GTSAM
         << "gtsam_std_x,gtsam_std_y,gtsam_std_z,gtsam_std_roll,gtsam_std_pitch,gtsam_std_yaw,"
         // Poses and RMSE
         << "ins_pose_r00,ins_pose_r01,ins_pose_r02,ins_pose_tx,"
         << "ins_pose_r10,ins_pose_r11,ins_pose_r12,ins_pose_ty,"
         << "ins_pose_r20,ins_pose_r21,ins_pose_r22,ins_pose_tz,"
         << "gtsam_pose_r00,gtsam_pose_r01,gtsam_pose_r02,gtsam_pose_tx,"
         << "gtsam_pose_r10,gtsam_pose_r11,gtsam_pose_r12,gtsam_pose_ty,"
         << "gtsam_pose_r20,gtsam_pose_r21,gtsam_pose_r22,gtsam_pose_tz,"
         << "pose_rmse\n";

    // Sort keys for ordered output
    std::vector<uint64_t> sorted_keys;
    sorted_keys.reserve(stats.size());
    for (const auto& pair : stats) {
        sorted_keys.push_back(pair.first);
    }
    std::sort(sorted_keys.begin(), sorted_keys.end());

    // Write data rows
    for (const auto& key : sorted_keys) {
        const auto& s = stats.at(key);
        file << std::fixed << std::setprecision(12);
        file << s.frame_id << "," << s.timestamp << "," << s.num_points << "," << s.alignment_time_ms << "," << s.ndt_iterations << ",";
        
        // Write rlla
        file << s.rlla.x() << "," << s.rlla.y() << "," << s.rlla.z() << ",";
        
        // Write all four Eigen vectors for standard deviations
        for (int i = 0; i < 6; ++i) file << s.ins_std_dev(i) << (i == 5 ? "" : ","); file << ",";
        for (int i = 0; i < 6; ++i) file << s.ins_scaled_std_dev(i) << (i == 5 ? "" : ","); file << ",";
        for (int i = 0; i < 6; ++i) file << s.lidar_std_dev(i) << (i == 5 ? "" : ","); file << ",";
        for (int i = 0; i < 6; ++i) file << s.gtsam_std_dev(i) << (i == 5 ? "" : ","); file << ",";
        
        // Eigen matrices (first 3 rows, 4 columns)
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c) file << s.ins_pose(r, c) << ((r==2 && c==3) ? "" : ","); file << ",";
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c) file << s.gtsam_pose(r, c) << ((r==2 && c==3) ? "" : ","); file << ",";
        
        file << s.pose_rmse << "\n";
    }
    file.close();
}
