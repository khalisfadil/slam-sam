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
//####################################################################################################
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
//####################################################################################################
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
//####################################################################################################
void writeCompasToFile(const CompasHashMap& compasArchive, const std::string& filename) {
    if (compasArchive.empty()) {
        std::cout << "No compass data to write." << std::endl;
        return;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing compass data: " << filename << std::endl;
        return;
    }

    // Write the complete CSV header row
    file << "frame_id,timestamp_20,latitude_20,longitude_20,altitude_20,"
         << "roll_20,pitch_20,yaw_20,"
         << "velocityNorth_20,velocityEast_20,velocityDown_20,"
         << "accelX_20,accelY_20,accelZ_20,"
         << "angularVelocityX_20,angularVelocityY_20,angularVelocityZ_20,"
         << "qw_20,qx_20,qy_20,qz_20,"
         << "gForce_20,GNSSFixStatus_20,"
         << "sigmaLatitude_20,sigmaLongitude_20,sigmaAltitude_20,"
         << "sigmaVelocityNorth_25,sigmaVelocityEast_25,sigmaVelocityDown_25,"
         << "sigmaRoll_26,sigmaPitch_26,sigmaYaw_26,"
         << "accelX_28,accelY_28,accelZ_28,"
         << "gyroX_28,gyroY_28,gyroZ_28,"
         << "magX_28,magY_28,magZ_28,"
         << "imuTemperature_28,pressure_28,pressureTemperature_28,"
         << "sigmaAccX_28,sigmaAccY_28,sigmaAccZ_28,"
         << "sigmaGyrX_28,sigmaGyrY_28,sigmaGyrZ_28,"
         << "biasAccX_28,biasAccY_28,biasAccZ_28,"
         << "biasGyrX_28,biasGyrY_28,biasGyrZ_28,"
         << "timestamp_29,latitude_29,longitude_29,altitude_29,"
         << "velocityNorth_29,velocityEast_29,velocityDown_29,"
         << "sigmaLatitude_29,sigmaLongitude_29,sigmaAltitude_29,"
         << "tilt_29,heading_29,sigmaTilt_29,sigmaHeading_29,"
         << "gnssFixStatus_29,dopplerVelocityValid_29,timeValid_29,externalGNSS_29,tiltValid_29,"
         << "SystemFailure_20,AccelerometerSensorFailure_20,GyroscopeSensorFailure_20,MagnetometerSensorFailure_20,"
         << "GNSSFailureSecondaryAntenna_20,GNSSFailurePrimaryAntenna_20,AccelerometerOverRange_20,"
         << "GyroscopeOverRange_20,MagnetometerOverRange_20,MinimumTemperatureAlarm_20,MaximumTemperatureAlarm_20,"
         << "GNSSAntennaConnectionBroken_20,DataOutputOverflowAlarm_20,OrientationFilterInitialised_20,"
         << "NavigationFilterInitialised_20,HeadingInitialised_20,UTCTimeInitialised_20,Event1_20,Event2_20,"
         << "InternalGNSSEnabled_20,DualAntennaHeadingActive_20,VelocityHeadingEnabled_20,GNSSFixInterrupted_20,"
         << "ExternalPositionActive_20,ExternalVelocityActive_20,ExternalHeadingActive_20\n";


    // Sort keys for ordered output
    std::vector<uint64_t> sorted_keys;
    sorted_keys.reserve(compasArchive.size());
    for (const auto& pair : compasArchive) {
        sorted_keys.push_back(pair.first);
    }
    std::sort(sorted_keys.begin(), sorted_keys.end());

    // Write data rows
    for (const auto& key : sorted_keys) {
        const auto& info = compasArchive.at(key);
        if (!info.ins) continue; // Safety check for null pointer
        const auto& f = *(info.ins);

        file << std::fixed 
             << info.frame_id << ","
             << std::setprecision(12) << f.timestamp_20 << "," << f.latitude_20 << "," << f.longitude_20 << "," << f.altitude_20 << ","
             << std::setprecision(6)
             << f.roll_20 << "," << f.pitch_20 << "," << f.yaw_20 << ","
             << f.velocityNorth_20 << "," << f.velocityEast_20 << "," << f.velocityDown_20 << ","
             << f.accelX_20 << "," << f.accelY_20 << "," << f.accelZ_20 << ","
             << f.angularVelocityX_20 << "," << f.angularVelocityY_20 << "," << f.angularVelocityZ_20 << ","
             << f.qw_20 << "," << f.qx_20 << "," << f.qy_20 << "," << f.qz_20 << ","
             << f.gForce_20 << "," << static_cast<int>(f.GNSSFixStatus_20) << ","
             << f.sigmaLatitude_20 << "," << f.sigmaLongitude_20 << "," << f.sigmaAltitude_20 << ","
             << f.sigmaVelocityNorth_25 << "," << f.sigmaVelocityEast_25 << "," << f.sigmaVelocityDown_25 << ","
             << f.sigmaRoll_26 << "," << f.sigmaPitch_26 << "," << f.sigmaYaw_26 << ","
             << f.accelX_28 << "," << f.accelY_28 << "," << f.accelZ_28 << ","
             << f.gyroX_28 << "," << f.gyroY_28 << "," << f.gyroZ_28 << ","
             << f.magX_28 << "," << f.magY_28 << "," << f.magZ_28 << ","
             << f.imuTemperature_28 << "," << f.pressure_28 << "," << f.pressureTemperature_28 << ","
             << f.sigmaAccX_28 << "," << f.sigmaAccY_28 << "," << f.sigmaAccZ_28 << ","
             << f.sigmaGyrX_28 << "," << f.sigmaGyrY_28 << "," << f.sigmaGyrZ_28 << ","
             << f.biasAccX_28 << "," << f.biasAccY_28 << "," << f.biasAccZ_28 << ","
             << f.biasGyrX_28 << "," << f.biasGyrY_28 << "," << f.biasGyrZ_28 << ","
             << std::setprecision(12)
             << f.timestamp_29 << "," << f.latitude_29 << "," << f.longitude_29 << "," << f.altitude_29 << ","
             << std::setprecision(6)
             << f.velocityNorth_29 << "," << f.velocityEast_29 << "," << f.velocityDown_29 << ","
             << f.sigmaLatitude_29 << "," << f.sigmaLongitude_29 << "," << f.sigmaAltitude_29 << ","
             << f.tilt_29 << "," << f.heading_29 << "," << f.sigmaTilt_29 << "," << f.sigmaHeading_29 << ","
             << static_cast<int>(f.gnssFixStatus_29) << "," << f.dopplerVelocityValid_29 << "," << f.timeValid_29 << "," << f.externalGNSS_29 << "," << f.tiltValid_29 << ","
             // Boolean flags
             << f.SystemFailure_20 << "," << f.AccelerometerSensorFailure_20 << "," << f.GyroscopeSensorFailure_20 << "," << f.MagnetometerSensorFailure_20 << ","
             << f.GNSSFailureSecondaryAntenna_20 << "," << f.GNSSFailurePrimaryAntenna_20 << "," << f.AccelerometerOverRange_20 << ","
             << f.GyroscopeOverRange_20 << "," << f.MagnetometerOverRange_20 << "," << f.MinimumTemperatureAlarm_20 << "," << f.MaximumTemperatureAlarm_20 << ","
             << f.GNSSAntennaConnectionBroken_20 << "," << f.DataOutputOverflowAlarm_20 << "," << f.OrientationFilterInitialised_20 << ","
             << f.NavigationFilterInitialised_20 << "," << f.HeadingInitialised_20 << "," << f.UTCTimeInitialised_20 << "," << f.Event1_20 << "," << f.Event2_20 << ","
             << f.InternalGNSSEnabled_20 << "," << f.DualAntennaHeadingActive_20 << "," << f.VelocityHeadingEnabled_20 << "," << f.GNSSFixInterrupted_20 << ","
             << f.ExternalPositionActive_20 << "," << f.ExternalVelocityActive_20 << "," << f.ExternalHeadingActive_20 
             << "\n";
    }
    file.close();
}
//####################################################################################################