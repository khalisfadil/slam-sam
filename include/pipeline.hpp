#pragma once

#include <iostream>
#include <thread>
#include <chrono>
#include <cstdint>

#include <queue>
#include <deque>
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
#include <gtsam/navigation/NavState.h>
#include <gtsam/base/Vector.h>
#include <gtsam/navigation/GPSFactor.h>

#include <pcl/visualization/pcl_visualizer.h> 
#include <pcl/filters/voxel_grid.h>          
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>           
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
template<typename T, typename = void>
struct has_clear_method : std::false_type {};

template<typename T>
struct has_clear_method<T, std::void_t<decltype(std::declval<T>().clear())>> : std::true_type {};

// Helper variable template
template<typename T>
inline constexpr bool has_clear_v = has_clear_method<T>::value;
//####################################################################################################
template <typename T>
class ObjectPool {
public:

    explicit ObjectPool(size_t initial_size = 0) {
        Initialize(initial_size);
    }

    ObjectPool(const ObjectPool&) = delete;
    ObjectPool& operator=(const ObjectPool&) = delete;
    ObjectPool(ObjectPool&&) = delete;
    ObjectPool& operator=(ObjectPool&&) = delete;

    void Initialize(size_t pool_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.clear(); // Clear existing pooled objects
        for (size_t i = 0; i < pool_size; ++i) {
            pool_.push_back(std::make_unique<T>());
        }
    }

    std::unique_ptr<T> Get() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pool_.empty()) {
            std::unique_ptr<T> ptr = std::move(pool_.front());
            pool_.pop_front();
            return ptr;
        }
        return std::make_unique<T>();
    }

    void Return(std::unique_ptr<T> ptr) {
        if (!ptr) {
            return; 
        }
        if constexpr (has_clear_v<T>) {
            // If it does, call it to reset the object's state.
            ptr->clear();
        }

        std::lock_guard<std::mutex> lock(mutex_);
        pool_.push_back(std::move(ptr));
    }

    size_t GetAvailableCount() const {
         std::lock_guard<std::mutex> lock(mutex_);
         return pool_.size();
    }

private:
    std::deque<std::unique_ptr<T>> pool_;
    mutable std::mutex mutex_;
};
//####################################################################################################
struct NdtEllipsoid {
    Eigen::Vector3d mean;
    Eigen::Matrix3d evecs; // Eigenvectors (Rotation)
    Eigen::Vector3d evals; // Eigenvalues (Variance / Radii)
};
//####################################################################################################
struct NdtVoxel {
    Eigen::Vector3d center;
    float resolution;
};
//####################################################################################################
template <typename PointT>
struct NdtExportData {
    std::vector<NdtEllipsoid> ellipsoids;
    std::vector<NdtVoxel> voxels;
    typename pcl::PointCloud<PointT> points; // A copy of the original points
};
//####################################################################################################
template <typename PointT, typename NDT_Type>
NdtExportData<PointT> extractNdtData(NDT_Type ndt,
        const typename pcl::PointCloud<PointT>::ConstPtr& map_cloud) {

    NdtExportData<PointT> export_data;

    // --- FIX: Workaround for pclomp const-correctness bug ---
    // We must use const_cast because ndt->getTargetCells() returns a
    // const TargetGrid, but the TargetGrid's methods 
    // .getLeaves() and .getMinPointPerVoxel() are NOT marked const.
    
    // 1. Define the type of the grid (from ndt_omp.h)
    using TargetGrid = pclomp::VoxelGridCovariance<PointT>;

    // 2. Get the const reference to the cells
    const TargetGrid& const_target_cells = ndt->getTargetCells();

    // 3. Cast away the const-ness to call the buggy non-const methods
    TargetGrid& non_const_target_cells = const_cast<TargetGrid&>(const_target_cells);

    // 4. Now call the non-const methods on the non-const reference
    auto leaves = non_const_target_cells.getLeaves();
    size_t min_points = non_const_target_cells.getMinPointPerVoxel();
    // --- End Fix ---

    float resolution = ndt->getResolution(); // This method is const-correct

    // Reserve space for a minor efficiency gain
    export_data.ellipsoids.reserve(leaves.size());
    export_data.voxels.reserve(leaves.size());

    // Loop through every leaf (voxel) in the NDT map
    for (auto const& [index, leaf] : leaves)
    {
        // A leaf must have enough points to be valid
        if (leaf.getPointCount() >= min_points)
        {
            // --- 1. Get Distribution Data (for Ellipsoids) ---
            export_data.ellipsoids.push_back(NdtEllipsoid{
                .mean = leaf.getMean(),
                .evecs = leaf.getEvecs(),
                .evals = leaf.getEvals()
            });

            // --- 2. Get Voxel Data (for Bounding Boxes) ---
            export_data.voxels.push_back(NdtVoxel{
                .center = leaf.getLeafCenter(),
                .resolution = resolution
            });
        }
    }

    // --- 3. Copy the original map points for reference ---
    // We make a copy so the data is fully contained in the struct
    if (map_cloud) { // Add a safety check for null pointer
        export_data.points = *map_cloud; 
    }

    std::cout << "Extracted " << export_data.ellipsoids.size() << " valid NDT leaves." << std::endl;
    std::cout << "Copied " << export_data.points.size() << " original map points." << std::endl;

    return export_data;
}
//####################################################################################################
template <typename PointT>
void writeNdtDataToFiles(const NdtExportData<PointT>& data,
                            const std::string& ellipsoid_filename,
                            const std::string& voxel_filename,
                            const std::string& points_filename) {
    // Create text files to export the data
    std::ofstream ellipsoid_file(ellipsoid_filename);
    std::ofstream voxel_file(voxel_filename);
    std::ofstream points_file(points_filename);

    // Set precision for clean output
    ellipsoid_file << std::fixed << std::setprecision(8);
    voxel_file << std::fixed << std::setprecision(8);
    points_file << std::fixed << std::setprecision(8);

    // --- 1. Write Ellipsoid Data ---
    for(const auto& ellipsoid : data.ellipsoids)
    {
        // Mean
        ellipsoid_file << ellipsoid.mean(0) << " " << ellipsoid.mean(1) << " " << ellipsoid.mean(2) << " ";
        
        // Eigenvectors (Row-major)
        const auto& evecs = ellipsoid.evecs;
        ellipsoid_file << evecs(0,0) << " " << evecs(0,1) << " " << evecs(0,2) << " ";
        ellipsoid_file << evecs(1,0) << " " << evecs(1,1) << " " << evecs(1,2) << " ";
        ellipsoid_file << evecs(2,0) << " " << evecs(2,1) << " " << evecs(2,2) << " ";

        // Eigenvalues
        const auto& evals = ellipsoid.evals;
        ellipsoid_file << evals(0) << " " << evals(1) << " " << evals(2) << "\n";
    }

    // --- 2. Write Voxel Data ---
    for(const auto& voxel : data.voxels)
    {
        voxel_file << voxel.center(0) << " " << voxel.center(1) << " " 
                   << voxel.center(2) << " " << voxel.resolution << "\n";
    }

    // --- 3. Write Point Data ---
    for(const auto& point : data.points)
    {
        points_file << point.x << " " << point.y << " " << point.z << "\n";
    }

    // Files are closed automatically when their streams go out of scope
    std::cout << "Successfully wrote NDT data to files." << std::endl;
}
//####################################################################################################
// void writeStatsToFile(const StatsHashMap& stats, const std::string& filename) {
//     if (stats.empty()) {
//         std::cout << "No stats to write." << std::endl;
//         return;
//     }

//     std::ofstream file(filename);
//     if (!file.is_open()) {
//         std::cerr << "Failed to open file for writing stats: " << filename << std::endl;
//         return;
//     }

//     // Write header with new rlla columns
//     file << "frame_id,timestamp,num_points,align_time_ms,ndt_iter,"
//          // Reference LLA
//          << "rlla_lat,rlla_lon,rlla_alt,"
//          // Unscaled INS
//          << "ins_unscaled_std_x,ins_unscaled_std_y,ins_unscaled_std_z,ins_unscaled_std_roll,ins_unscaled_std_pitch,ins_unscaled_std_yaw,"
//          // Scaled INS
//          << "ins_scaled_std_x,ins_scaled_std_y,ins_scaled_std_z,ins_scaled_std_roll,ins_scaled_std_pitch,ins_scaled_std_yaw,"
//          // Lidar
//          << "lidar_std_x,lidar_std_y,lidar_std_z,lidar_std_roll,lidar_std_pitch,lidar_std_yaw,"
//          // GTSAM
//          << "gtsam_std_x,gtsam_std_y,gtsam_std_z,gtsam_std_roll,gtsam_std_pitch,gtsam_std_yaw,"
//          // Poses and RMSE
//          << "ins_pose_r00,ins_pose_r01,ins_pose_r02,ins_pose_tx,"
//          << "ins_pose_r10,ins_pose_r11,ins_pose_r12,ins_pose_ty,"
//          << "ins_pose_r20,ins_pose_r21,ins_pose_r22,ins_pose_tz,"
//          << "gtsam_pose_r00,gtsam_pose_r01,gtsam_pose_r02,gtsam_pose_tx,"
//          << "gtsam_pose_r10,gtsam_pose_r11,gtsam_pose_r12,gtsam_pose_ty,"
//          << "gtsam_pose_r20,gtsam_pose_r21,gtsam_pose_r22,gtsam_pose_tz,"
//          << "pose_rmse\n";

//     // Sort keys for ordered output
//     std::vector<uint64_t> sorted_keys;
//     sorted_keys.reserve(stats.size());
//     for (const auto& pair : stats) {
//         sorted_keys.push_back(pair.first);
//     }
//     std::sort(sorted_keys.begin(), sorted_keys.end());

//     // Write data rows
//     for (const auto& key : sorted_keys) {
//         const auto& s = stats.at(key);
//         file << std::fixed << std::setprecision(12);
//         file << s.frame_id << "," << s.timestamp << "," << s.num_points << "," << s.alignment_time_ms << "," << s.ndt_iterations << ",";
        
//         // Write rlla
//         file << s.rlla.x() << "," << s.rlla.y() << "," << s.rlla.z() << ",";
        
//         // Write all four Eigen vectors for standard deviations
//         for (int i = 0; i < 6; ++i) file << s.ins_std_dev(i) << (i == 5 ? "" : ","); file << ",";
//         for (int i = 0; i < 6; ++i) file << s.ins_scaled_std_dev(i) << (i == 5 ? "" : ","); file << ",";
//         for (int i = 0; i < 6; ++i) file << s.lidar_std_dev(i) << (i == 5 ? "" : ","); file << ",";
//         for (int i = 0; i < 6; ++i) file << s.gtsam_std_dev(i) << (i == 5 ? "" : ","); file << ",";
        
//         // Eigen matrices (first 3 rows, 4 columns)
//         for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c) file << s.ins_pose(r, c) << ((r==2 && c==3) ? "" : ","); file << ",";
//         for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c) file << s.gtsam_pose(r, c) << ((r==2 && c==3) ? "" : ","); file << ",";
        
//         file << s.pose_rmse << "\n";
//     }
//     file.close();
// }
// //####################################################################################################
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
         << "timestamp_29,latitude_29,longitude_29,altitude_29,"
         << "velocityNorth_29,velocityEast_29,velocityDown_29,"
         << "sigmaLatitude_29,sigmaLongitude_29,sigmaAltitude_29,"
         << "tilt_29,heading_29,sigmaTilt_29,sigmaHeading_29,"
         << "GNSSFixStatus_29,dopplerVelocityValid_29,timeValid_29,externalGNSS_29,tiltValid_29,"
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
             << std::setprecision(12)
             << f.timestamp_29 << "," << f.latitude_29 << "," << f.longitude_29 << "," << f.altitude_29 << ","
             << std::setprecision(6)
             << f.velocityNorth_29 << "," << f.velocityEast_29 << "," << f.velocityDown_29 << ","
             << f.sigmaLatitude_29 << "," << f.sigmaLongitude_29 << "," << f.sigmaAltitude_29 << ","
             << f.tilt_29 << "," << f.heading_29 << "," << f.sigmaTilt_29 << "," << f.sigmaHeading_29 << ","
             << static_cast<int>(f.GNSSFixStatus_29) << "," << f.dopplerVelocityValid_29 << "," << f.timeValid_29 << "," << f.externalGNSS_29 << "," << f.tiltValid_29 << ","
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
// //####################################################################################################