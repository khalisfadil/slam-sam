#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/linear/NoiseModel.h>

#include <map.hpp>
#include <algorithm>
#include <cstdint> 
#include <memory>

// %             ... struct for parameter
struct CompFrame {
    // --- FROM CompFrameID20 ---
    bool valid_20;
    bool valid_25;
    bool valid_26;
    bool valid_28;
    bool valid_29;

    bool SystemFailure_20;
    bool AccelerometerSensorFailure_20;
    bool GyroscopeSensorFailure_20;
    bool MagnetometerSensorFailure_20;
    bool GNSSFailureSecondaryAntenna_20;
    bool GNSSFailurePrimaryAntenna_20;
    bool AccelerometerOverRange_20;
    bool GyroscopeOverRange_20;
    bool MagnetometerOverRange_20;
    bool MinimumTemperatureAlarm_20;
    bool MaximumTemperatureAlarm_20;
    bool GNSSAntennaConnectionBroken_20;
    bool DataOutputOverflowAlarm_20;
    bool OrientationFilterInitialised_20;
    bool NavigationFilterInitialised_20;
    bool HeadingInitialised_20;
    bool UTCTimeInitialised_20;
    uint8_t GNSSFixStatus_20;
    bool Event1_20;
    bool Event2_20;
    bool InternalGNSSEnabled_20;
    bool DualAntennaHeadingActive_20;
    bool VelocityHeadingEnabled_20;
    bool GNSSFixInterrupted_20;
    bool ExternalPositionActive_20;
    bool ExternalVelocityActive_20;
    bool ExternalHeadingActive_20;
    double timestamp_20;
    double latitude_20;
    double longitude_20;
    double altitude_20;
    float velocityNorth_20;
    float velocityEast_20;
    float velocityDown_20;
    float accelX_20;
    float accelY_20;
    float accelZ_20;
    float gForce_20;
    float roll_20;
    float pitch_20;
    float yaw_20;
    float qw_20;
    float qx_20;
    float qy_20;
    float qz_20;
    float angularVelocityX_20;
    float angularVelocityY_20;
    float angularVelocityZ_20;
    float sigmaLatitude_20;
    float sigmaLongitude_20;
    float sigmaAltitude_20;

    // --- FROM CompFrameID25 ---
    float sigmaVelocityNorth_25;
    float sigmaVelocityEast_25;
    float sigmaVelocityDown_25;

    // --- FROM CompFrameID26 ---
    float sigmaRoll_26;
    float sigmaPitch_26;
    float sigmaYaw_26;

    // --- FROM CompFrameID28 ---
    float accelX_28;
    float accelY_28;
    float accelZ_28;
    float gyroX_28;
    float gyroY_28;
    float gyroZ_28;
    float magX_28;
    float magY_28;
    float magZ_28;
    float imuTemperature_28;
    float pressure_28;
    float pressureTemperature_28;

    // --- FROM DataFrameID29 ---
    double timestamp_29;
    double latitude_29;
    double longitude_29;
    double altitude_29;
    float velocityNorth_29;
    float velocityEast_29;
    float velocityDown_29;
    float sigmaLatitude_29;
    float sigmaLongitude_29;
    float sigmaAltitude_29;
    float tilt_29;
    float heading_29;
    float sigmaTilt_29;
    float sigmaHeading_29;
    uint8_t GNSSFixStatus_29;
    bool dopplerVelocityValid_29;
    bool timeValid_29;
    bool externalGNSS_29;
    bool tiltValid_29;

    /**
     * @brief Default constructor that initializes all member variables to a default state.
     */
    CompFrame() {
        // Clear ID20
        valid_20 = false; valid_25 = false; valid_26 = false; valid_28 = false; valid_29 = false;
        SystemFailure_20 = false; AccelerometerSensorFailure_20 = false; GyroscopeSensorFailure_20 = false;
        MagnetometerSensorFailure_20 = false; GNSSFailureSecondaryAntenna_20 = false; GNSSFailurePrimaryAntenna_20 = false;
        AccelerometerOverRange_20 = false; GyroscopeOverRange_20 = false; MagnetometerOverRange_20 = false;
        MinimumTemperatureAlarm_20 = false; MaximumTemperatureAlarm_20 = false; GNSSAntennaConnectionBroken_20 = false;
        DataOutputOverflowAlarm_20 = false; OrientationFilterInitialised_20 = false; NavigationFilterInitialised_20 = false;
        HeadingInitialised_20 = false; UTCTimeInitialised_20 = false; GNSSFixStatus_20 = 0; Event1_20 = false;
        Event2_20 = false; InternalGNSSEnabled_20 = false; DualAntennaHeadingActive_20 = false;
        VelocityHeadingEnabled_20 = false; GNSSFixInterrupted_20 = false; ExternalPositionActive_20 = false;
        ExternalVelocityActive_20 = false; ExternalHeadingActive_20 = false; timestamp_20 = 0.0; latitude_20 = 0.0;
        longitude_20 = 0.0; altitude_20 = 0.0; velocityNorth_20 = 0.0f; velocityEast_20 = 0.0f;
        velocityDown_20 = 0.0f; accelX_20 = 0.0f; accelY_20 = 0.0f; accelZ_20 = 0.0f; gForce_20 = 0.0f;
        roll_20 = 0.0f; pitch_20 = 0.0f; yaw_20 = 0.0f;
        qw_20 = 1.0f; qx_20 = 0.0f; qy_20 = 0.0f; qz_20 = 0.0f; // Identity quaternion
        angularVelocityX_20 = 0.0f;
        angularVelocityY_20 = 0.0f; angularVelocityZ_20 = 0.0f; sigmaLatitude_20 = 0.0f;
        sigmaLongitude_20 = 0.0f; sigmaAltitude_20 = 0.0f;

        // Clear ID25
        sigmaVelocityNorth_25 = 0.0f; sigmaVelocityEast_25 = 0.0f; sigmaVelocityDown_25 = 0.0f;

        // Clear ID26
        sigmaRoll_26 = 0.0f; sigmaPitch_26 = 0.0f; sigmaYaw_26 = 0.0f;

        // Clear ID28
        accelX_28 = 0.0f; accelY_28 = 0.0f; accelZ_28 = 0.0f; gyroX_28 = 0.0f; gyroY_28 = 0.0f;
        gyroZ_28 = 0.0f; magX_28 = 0.0f; magY_28 = 0.0f; magZ_28 = 0.0f; imuTemperature_28 = 0.0f;
        pressure_28 = 0.0f; pressureTemperature_28 = 0.0f;

        // Clear ID29
        timestamp_29 = 0.0; latitude_29 = 0.0; longitude_29 = 0.0; altitude_29 = 0.0; velocityNorth_29 = 0.0f;
        velocityEast_29 = 0.0f; velocityDown_29 = 0.0f; sigmaLatitude_29 = 0.0f; sigmaLongitude_29 = 0.0f;
        sigmaAltitude_29 = 0.0f; tilt_29 = 0.0f; heading_29 = 0.0f; sigmaTilt_29 = 0.0f; sigmaHeading_29 = 0.0f;
        GNSSFixStatus_29 = 0; dopplerVelocityValid_29 = false; timeValid_29 = false; externalGNSS_29 = false;
        tiltValid_29 = false;
    }

    /**
     * @brief Resets an existing object to its default state by assigning a new, default-constructed object.
     */
    void clear() {
        *this = CompFrame();
    }

    bool isValid() const {
        return this->valid_20 && this->valid_25 && this->valid_26 && this->valid_28 && this->valid_29;
    }

    /**
     * @brief Creates an interpolated CompFrame between two given frames.
     * @param a The starting frame (corresponding to t=0).
     * @param b The ending frame (corresponding to t=1).
     * @param t The interpolation factor, which will be clamped to the [0, 1] range.
     * @return A new CompFrame object with interpolated values.
     */
    [[nodiscard]] CompFrame linearInterpolate(const CompFrame& a, const CompFrame& b, double t) const {
        CompFrame result;
        const auto clamped_t = static_cast<float>(std::max(0.0, std::min(1.0, t)));

        // --- STRATEGY 1: Linear Interpolation (continuous numeric values) ---
        // ID20
        result.timestamp_20 = a.timestamp_20 + clamped_t * (b.timestamp_20 - a.timestamp_20);
        result.latitude_20 = a.latitude_20 + clamped_t * (b.latitude_20 - a.latitude_20);
        result.longitude_20 = a.longitude_20 + clamped_t * (b.longitude_20 - a.longitude_20);
        result.altitude_20 = a.altitude_20 + clamped_t * (b.altitude_20 - a.altitude_20);
        result.velocityNorth_20 = a.velocityNorth_20 + clamped_t * (b.velocityNorth_20 - a.velocityNorth_20);
        result.velocityEast_20 = a.velocityEast_20 + clamped_t * (b.velocityEast_20 - a.velocityEast_20);
        result.velocityDown_20 = a.velocityDown_20 + clamped_t * (b.velocityDown_20 - a.velocityDown_20);
        result.accelX_20 = a.accelX_20 + clamped_t * (b.accelX_20 - a.accelX_20);
        result.accelY_20 = a.accelY_20 + clamped_t * (b.accelY_20 - a.accelY_20);
        result.accelZ_20 = a.accelZ_20 + clamped_t * (b.accelZ_20 - a.accelZ_20);
        result.gForce_20 = a.gForce_20 + clamped_t * (b.gForce_20 - a.gForce_20);
        result.roll_20 = a.roll_20 + clamped_t * (b.roll_20 - a.roll_20);
        result.pitch_20 = a.pitch_20 + clamped_t * (b.pitch_20 - a.pitch_20);
        result.yaw_20 = a.yaw_20 + clamped_t * (b.yaw_20 - a.yaw_20);
        result.angularVelocityX_20 = a.angularVelocityX_20 + clamped_t * (b.angularVelocityX_20 - a.angularVelocityX_20);
        result.angularVelocityY_20 = a.angularVelocityY_20 + clamped_t * (b.angularVelocityY_20 - a.angularVelocityY_20);
        result.angularVelocityZ_20 = a.angularVelocityZ_20 + clamped_t * (b.angularVelocityZ_20 - a.angularVelocityZ_20);
        result.sigmaLatitude_20 = a.sigmaLatitude_20 + clamped_t * (b.sigmaLatitude_20 - a.sigmaLatitude_20);
        result.sigmaLongitude_20 = a.sigmaLongitude_20 + clamped_t * (b.sigmaLongitude_20 - a.sigmaLongitude_20);
        result.sigmaAltitude_20 = a.sigmaAltitude_20 + clamped_t * (b.sigmaAltitude_20 - a.sigmaAltitude_20);
        // ID25
        result.sigmaVelocityNorth_25 = a.sigmaVelocityNorth_25 + clamped_t * (b.sigmaVelocityNorth_25 - a.sigmaVelocityNorth_25);
        result.sigmaVelocityEast_25 = a.sigmaVelocityEast_25 + clamped_t * (b.sigmaVelocityEast_25 - a.sigmaVelocityEast_25);
        result.sigmaVelocityDown_25 = a.sigmaVelocityDown_25 + clamped_t * (b.sigmaVelocityDown_25 - a.sigmaVelocityDown_25);
        // ID26
        result.sigmaRoll_26 = a.sigmaRoll_26 + clamped_t * (b.sigmaRoll_26 - a.sigmaRoll_26);
        result.sigmaPitch_26 = a.sigmaPitch_26 + clamped_t * (b.sigmaPitch_26 - a.sigmaPitch_26);
        result.sigmaYaw_26 = a.sigmaYaw_26 + clamped_t * (b.sigmaYaw_26 - a.sigmaYaw_26);
        // ID28
        result.accelX_28 = a.accelX_28 + clamped_t * (b.accelX_28 - a.accelX_28);
        result.accelY_28 = a.accelY_28 + clamped_t * (b.accelY_28 - a.accelY_28);
        result.accelZ_28 = a.accelZ_28 + clamped_t * (b.accelZ_28 - a.accelZ_28);
        result.gyroX_28 = a.gyroX_28 + clamped_t * (b.gyroX_28 - a.gyroX_28);
        result.gyroY_28 = a.gyroY_28 + clamped_t * (b.gyroY_28 - a.gyroY_28);
        result.gyroZ_28 = a.gyroZ_28 + clamped_t * (b.gyroZ_28 - a.gyroZ_28);
        result.magX_28 = a.magX_28 + clamped_t * (b.magX_28 - a.magX_28);
        result.magY_28 = a.magY_28 + clamped_t * (b.magY_28 - a.magY_28);
        result.magZ_28 = a.magZ_28 + clamped_t * (b.magZ_28 - a.magZ_28);
        result.imuTemperature_28 = a.imuTemperature_28 + clamped_t * (b.imuTemperature_28 - a.imuTemperature_28);
        result.pressure_28 = a.pressure_28 + clamped_t * (b.pressure_28 - a.pressure_28);
        result.pressureTemperature_28 = a.pressureTemperature_28 + clamped_t * (b.pressureTemperature_28 - a.pressureTemperature_28);

        // ID29
        result.timestamp_29 = a.timestamp_29 + clamped_t * (b.timestamp_29 - a.timestamp_29);
        result.latitude_29 = a.latitude_29 + clamped_t * (b.latitude_29 - a.latitude_29);
        result.longitude_29 = a.longitude_29 + clamped_t * (b.longitude_29 - a.longitude_29);
        result.altitude_29 = a.altitude_29 + clamped_t * (b.altitude_29 - a.altitude_29);
        result.velocityNorth_29 = a.velocityNorth_29 + clamped_t * (b.velocityNorth_29 - a.velocityNorth_29);
        result.velocityEast_29 = a.velocityEast_29 + clamped_t * (b.velocityEast_29 - a.velocityEast_29);
        result.velocityDown_29 = a.velocityDown_29 + clamped_t * (b.velocityDown_29 - a.velocityDown_29);
        result.sigmaLatitude_29 = a.sigmaLatitude_29 + clamped_t * (b.sigmaLatitude_29 - a.sigmaLatitude_29);
        result.sigmaLongitude_29 = a.sigmaLongitude_29 + clamped_t * (b.sigmaLongitude_29 - a.sigmaLongitude_29);
        result.sigmaAltitude_29 = a.sigmaAltitude_29 + clamped_t * (b.sigmaAltitude_29 - a.sigmaAltitude_29);
        result.tilt_29 = a.tilt_29 + clamped_t * (b.tilt_29 - a.tilt_29);
        result.heading_29 = a.heading_29 + clamped_t * (b.heading_29 - a.heading_29);
        result.sigmaTilt_29 = a.sigmaTilt_29 + clamped_t * (b.sigmaTilt_29 - a.sigmaTilt_29);
        result.sigmaHeading_29 = a.sigmaHeading_29 + clamped_t * (b.sigmaHeading_29 - a.sigmaHeading_29);

        // --- STRATEGY 2: Spherical Linear Interpolation (Slerp) ---
        Eigen::Quaternionf q_a(a.qw_20, a.qx_20, a.qy_20, a.qz_20);
        Eigen::Quaternionf q_b(b.qw_20, b.qx_20, b.qy_20, b.qz_20);
        Eigen::Quaternionf q_result = q_a.slerp(clamped_t, q_b);
        result.qw_20 = q_result.w();
        result.qx_20 = q_result.x();
        result.qy_20 = q_result.y();
        result.qz_20 = q_result.z();

        // --- STRATEGY 3: Logical OR (for failures, alarms, and transient events) ---
        result.SystemFailure_20 = a.SystemFailure_20 || b.SystemFailure_20;
        result.AccelerometerSensorFailure_20 = a.AccelerometerSensorFailure_20 || b.AccelerometerSensorFailure_20;
        result.GyroscopeSensorFailure_20 = a.GyroscopeSensorFailure_20 || b.GyroscopeSensorFailure_20;
        result.MagnetometerSensorFailure_20 = a.MagnetometerSensorFailure_20 || b.MagnetometerSensorFailure_20;
        result.GNSSFailureSecondaryAntenna_20 = a.GNSSFailureSecondaryAntenna_20 || b.GNSSFailureSecondaryAntenna_20;
        result.GNSSFailurePrimaryAntenna_20 = a.GNSSFailurePrimaryAntenna_20 || b.GNSSFailurePrimaryAntenna_20;
        result.AccelerometerOverRange_20 = a.AccelerometerOverRange_20 || b.AccelerometerOverRange_20;
        result.GyroscopeOverRange_20 = a.GyroscopeOverRange_20 || b.GyroscopeOverRange_20;
        result.MagnetometerOverRange_20 = a.MagnetometerOverRange_20 || b.MagnetometerOverRange_20;
        result.MinimumTemperatureAlarm_20 = a.MinimumTemperatureAlarm_20 || b.MinimumTemperatureAlarm_20;
        result.MaximumTemperatureAlarm_20 = a.MaximumTemperatureAlarm_20 || b.MaximumTemperatureAlarm_20;
        result.GNSSAntennaConnectionBroken_20 = a.GNSSAntennaConnectionBroken_20 || b.GNSSAntennaConnectionBroken_20;
        result.DataOutputOverflowAlarm_20 = a.DataOutputOverflowAlarm_20 || b.DataOutputOverflowAlarm_20;
        result.Event1_20 = a.Event1_20 || b.Event1_20;
        result.Event2_20 = a.Event2_20 || b.Event2_20;
        result.GNSSFixInterrupted_20 = a.GNSSFixInterrupted_20 || b.GNSSFixInterrupted_20;

        // --- STRATEGY 4: Logical AND (for initialization and stable states) ---
        result.OrientationFilterInitialised_20 = a.OrientationFilterInitialised_20 && b.OrientationFilterInitialised_20;
        result.NavigationFilterInitialised_20 = a.NavigationFilterInitialised_20 && b.NavigationFilterInitialised_20;
        result.HeadingInitialised_20 = a.HeadingInitialised_20 && b.HeadingInitialised_20;
        result.UTCTimeInitialised_20 = a.UTCTimeInitialised_20 && b.UTCTimeInitialised_20;
        result.InternalGNSSEnabled_20 = a.InternalGNSSEnabled_20 && b.InternalGNSSEnabled_20;
        result.DualAntennaHeadingActive_20 = a.DualAntennaHeadingActive_20 && b.DualAntennaHeadingActive_20;
        result.VelocityHeadingEnabled_20 = a.VelocityHeadingEnabled_20 && b.VelocityHeadingEnabled_20;
        result.ExternalPositionActive_20 = a.ExternalPositionActive_20 && b.ExternalPositionActive_20;
        result.ExternalVelocityActive_20 = a.ExternalVelocityActive_20 && b.ExternalVelocityActive_20;
        result.ExternalHeadingActive_20 = a.ExternalHeadingActive_20 && b.ExternalHeadingActive_20;
        result.dopplerVelocityValid_29 = a.dopplerVelocityValid_29 && b.dopplerVelocityValid_29;
        result.timeValid_29 = a.timeValid_29 && b.timeValid_29;
        result.externalGNSS_29 = a.externalGNSS_29 && b.externalGNSS_29;
        result.tiltValid_29 = a.tiltValid_29 && b.tiltValid_29;

        // --- STRATEGY 5: Nearest Neighbor (for discrete status codes) ---
        result.GNSSFixStatus_20 = (clamped_t < 0.5) ? a.GNSSFixStatus_20 : b.GNSSFixStatus_20;
        result.GNSSFixStatus_29 = (clamped_t < 0.5) ? a.GNSSFixStatus_29 : b.GNSSFixStatus_29;

        return result;
    }
};
// %            ... struct representing single 3d point data
struct VisualizationData {
    std::shared_ptr<gtsam::Values> poses;
    std::shared_ptr<PointsHashMap> points;
    // std::shared_ptr<StateHashMap> insposes;
    std::shared_ptr<PoseHashMap> insposes;
};
// %            ... struct representing single 3d point data
struct PCLPointCloud{
    uint16_t frame_id = 0;
    pcl::PointCloud<pcl::PointXYZI> pointsBody;               
    std::vector<float> pointsAlpha;
    std::vector<double> pointsTimestamp;                                             
};
// %            ... struct representing a lidar data and its encapsulated data of imu and position
struct FrameData{
    double timestamp;                                               // evaluation timestamp
    PCLPointCloud points;
    std::vector<CompFrame> ins;
};
// %             ... struct for parameter
struct LidarFrame {
    uint16_t frame_id = 0;
    double timestamp = 0.0; // Current timestamp, unix timestamp (PTP sync)
    double timestamp_end = 0.0; // End timestamp of current frame
    double interframe_timedelta = 0.0; // Time difference between first point in current frame and last point in last frame
    uint32_t numberpoints = 0;

    std::vector<float, Eigen::aligned_allocator<float>> x; // X coordinates
    std::vector<float, Eigen::aligned_allocator<float>> y; // Y coordinates
    std::vector<float, Eigen::aligned_allocator<float>> z; // Z coordinates
    std::vector<uint16_t> c_id; // Channel indices
    std::vector<uint16_t> m_id; // Measurement indices
    std::vector<double, Eigen::aligned_allocator<double>> timestamp_points; // Absolute timestamps
    std::vector<float, Eigen::aligned_allocator<float>> relative_timestamp; // Relative timestamps
    std::vector<uint16_t> reflectivity; // Reflectivity values
    std::vector<uint16_t> signal; // Signal strengths
    std::vector<uint16_t> nir; // NIR values

    void reserve(size_t size) {
        x.reserve(size);
        y.reserve(size);
        z.reserve(size);
        c_id.reserve(size);
        m_id.reserve(size);
        timestamp_points.reserve(size);
        relative_timestamp.reserve(size);
        reflectivity.reserve(size);
        signal.reserve(size);
        nir.reserve(size);
    }

    void clear() {
        frame_id = 0;
        timestamp = 0.0;
        timestamp_end = 0.0;
        interframe_timedelta = 0.0;
        numberpoints = 0;
        x.clear();
        y.clear();
        z.clear();
        c_id.clear();
        m_id.clear();
        timestamp_points.clear();
        relative_timestamp.clear();
        reflectivity.clear();
        signal.clear();
        nir.clear();
    }

    void swap(LidarFrame& other) noexcept {
        std::swap(frame_id, other.frame_id);
        std::swap(timestamp, other.timestamp);
        std::swap(timestamp_end, other.timestamp_end);
        std::swap(interframe_timedelta, other.interframe_timedelta);
        std::swap(numberpoints, other.numberpoints);

        // Swap the vectors. This is the key to efficiency!
        x.swap(other.x);
        y.swap(other.y);
        z.swap(other.z);
        c_id.swap(other.c_id);
        m_id.swap(other.m_id);
        timestamp_points.swap(other.timestamp_points);
        relative_timestamp.swap(other.relative_timestamp);
        reflectivity.swap(other.reflectivity);
        signal.swap(other.signal);
        nir.swap(other.nir);
    }

    [[nodiscard]] PCLPointCloud toPCLPointCloud() const {
        PCLPointCloud pointcloud;
        pointcloud.pointsBody.reserve(numberpoints);
        pointcloud.pointsAlpha.reserve(numberpoints);
        pointcloud.pointsTimestamp.reserve(numberpoints);

        const double frame_duration = this->timestamp_end - this->timestamp;
        pointcloud.frame_id = this->frame_id;
        for (size_t i = 0; i < numberpoints; ++i) {
            pcl::PointXYZI point;
            point.x = this->x[i];
            point.y = this->y[i];
            point.z = this->z[i];
            point.intensity = static_cast<float>(this->reflectivity[i]); // Use reflectivity for intensity
            // Add to pointsBody (raw sensor coordinates)
            pointcloud.pointsBody.push_back(point);
            // Calculate alpha (interpolation factor)
            float alpha = 0.0f;
            if (frame_duration > 0.0) {
                const double elapsed_time = this->timestamp_points[i] - this->timestamp;
                alpha = static_cast<float>(std::max(0.0, std::min(1.0, elapsed_time / frame_duration)));
            }
            pointcloud.pointsAlpha.push_back(alpha);
            // Add absolute timestamp
            pointcloud.pointsTimestamp.push_back(this->timestamp_points[i]);
        }
        return pointcloud;
    }
};

struct KeyCompasInfo {
    std::unique_ptr<CompFrame> ins; // Tb2m: body-to-map transformation in NED frame
    double timestamp;
    uint64_t frame_id = 0;
};
using CompasHashMap = tsl::robin_map<uint64_t, KeyCompasInfo, KeyframeHash>;







