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

void manualTransformPointCloud_RowBased(
        const pcl::PointCloud<pcl::PointXYZI>& cloud_in,
        pcl::PointCloud<pcl::PointXYZI>& cloud_out,
        const Eigen::Matrix4f& transform)
    {
        cloud_out.clear();
        cloud_out.resize(cloud_in.size());

        // Extract the 3x3 rotation matrix (R).
        Eigen::Matrix3f rotation_matrix = transform.block<3, 3>(0, 0);
        // Extract the translation vector and represent it as a 1x3 row vector.
        Eigen::RowVector3f translation_row_vector = transform.block<3, 1>(0, 3).transpose();

        for (size_t i = 0; i < cloud_in.size(); ++i) {
            const auto& p_in = cloud_in.points[i];

            // Represent the input point as a 1x3 Eigen Row Vector.
            Eigen::RowVector3f p_in_row_vec(p_in.x, p_in.y, p_in.z);

            // --- THE CORE ROW-BASED TRANSFORMATION ---
            // Mimics MATLAB: p' = p * R' + t
            Eigen::RowVector3f p_out_row_vec = p_in_row_vec * rotation_matrix.transpose() + translation_row_vector;

            auto& p_out = cloud_out.points[i];
            p_out.x = -p_out_row_vec.x();
            p_out.y = p_out_row_vec.y();
            p_out.z = p_out_row_vec.z();
            p_out.intensity = p_in.intensity;
        }
    }