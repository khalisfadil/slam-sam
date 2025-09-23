#pragma once

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/linear/NoiseModel.h>

#include <map.hpp>

#include <cstdint> 
#include <memory>

// %            ... struct representing single 3d point data
struct VisualizationData {
    std::shared_ptr<gtsam::Values> poses;
    std::shared_ptr<PointsHashMap> points;
};
// %            ... struct representing single 3d point data
struct PCLPointCloud{
    uint16_t frame_id = 0;
    pcl::PointCloud<pcl::PointXYZI> pointsBody;               
    std::vector<float> pointsAlpha;
    std::vector<double> pointsTimestamp;                                             
};
// %            ... struct representing single frame imu data
struct ImuData{
    Eigen::Vector3f acc = Eigen::Vector3f::Zero();                  // acceleration x,y,z in IMU sensor frame                                                     [m/(s*s)]
    Eigen::Vector3f gyr = Eigen::Vector3f::Zero();                  // angular rate around x-axis,y-axis,z-axis in IMU sensor frame                               [rad/s]
    Eigen::Vector3f accStdDev = Eigen::Vector3f::Zero();            // standard deviation for acceleration x,y,z in IMU sensor frame                              [m/(s*s)]
    Eigen::Vector3f gyrStdDev = Eigen::Vector3f::Zero();            // standard deviation for angular rate around x-axis,y-axis,z-axis in IMU sensor frame        [rad/s] 
    double timestamp = 0.0;                                         // absolute timestamp of the data                                                             [s]
};
// %            ... struct representing single frame in position data (GPS,GNSS,INS, etc..)
struct PositionData{
    Eigen::Vector3d pose = Eigen::Vector3d::Zero();                 // position latitude[rad],longitude[rad],altitude[m] in sensor frame                          ([rad],[rad],[m])
    Eigen::Vector3f euler = Eigen::Vector3d::Zero();
    Eigen::Quaternionf orientation = Eigen::Quaternionf::Identity(); // orientation as quaternion (w, x, y, z)
    Eigen::Vector3f poseStdDev = Eigen::Vector3f::Zero();           // standard deviation in north,east,down in sensor frame   
    Eigen::Vector3f eulerStdDev = Eigen::Vector3f::Zero();          // standard deviation in north,east,down in sensor frame                                   [m]
    double timestamp = 0.0;                                         // absolute timestamp of the data                                                             [s]
};
// %            ... struct representing a lidar data and its encapsulated data of imu and position
struct FrameData{
    double timestamp;                                               // evaluation timestamp
    PCLPointCloud points;
    std::vector<ImuData> imu;
    std::vector<PositionData> position;
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
        numberpoints = 0;
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
// %             ... struct for parameter
struct CompFrame {
    // System Status (Bytes 6-7)
    bool SystemFailure = false;
    bool AccelerometerSensorFailure = false;
    bool GyroscopeSensorFailure = false;
    bool MagnetometerSensorFailure = false;
    bool GNSSFailureSecondaryAntenna = false;
    bool GNSSFailurePrimaryAntenna = false;
    bool AccelerometerOverRange = false;
    bool GyroscopeOverRange = false;
    bool MagnetometerOverRange = false;
    bool MinimumTemperatureAlarm = false;
    bool MaximumTemperatureAlarm = false;
    bool GNSSAntennaConnectionBroken = false;
    bool DataOutputOverflowAlarm = false;

    // Filter Status (Bytes 8-9)
    bool OrientationFilterInitialised = false;
    bool NavigationFilterInitialised = false;
    bool HeadingInitialised = false;
    bool UTCTimeInitialised = false;
    uint8_t GNSSFixStatus = 0; // 3-bit field (0-7) for GNSS fix type
    bool Event1 = false;
    bool Event2 = false;
    bool InternalGNSSEnabled = false;
    bool DualAntennaHeadingActive = false;
    bool VelocityHeadingEnabled = false;
    bool GNSSFixInterrupted = false;
    bool ExternalPositionActive = false;
    bool ExternalVelocityActive = false;
    bool ExternalHeadingActive = false;

    // Navigation Data
    double timestamp = 0.0; // UTC Timestamp in seconds
    double latitude = 0.0;  // Latitude in radians
    double longitude = 0.0; // Longitude in radians
    double altitude = 0.0;  // Altitude in meters
    float velocityNorth = 0.0f; // Velocity north in m/s
    float velocityEast = 0.0f;  // Velocity east in m/s
    float velocityDown = 0.0f;  // Velocity down in m/s
    float accelX = 0.0f;        // Body acceleration X in m/s^2
    float accelY = 0.0f;        // Body acceleration Y in m/s^2
    float accelZ = 0.0f;        // Body acceleration Z in m/s^2
    float gForce = 0.0f;        // G force in g
    float roll = 0.0f;
    float pitch = 0.0f;
    float yaw = 0.0f;
    Eigen::Quaternionf orientation = Eigen::Quaternionf::Identity();
    float angularVelocityX = 0.0f; // Angular velocity X in rad/s
    float angularVelocityY = 0.0f; // Angular velocity Y in rad/s
    float angularVelocityZ = 0.0f; // Angular velocity Z in rad/s
    float sigmaLatitude = 0.0f;    // Latitude standard deviation in meters
    float sigmaLongitude = 0.0f;   // Longitude standard deviation in meters
    float sigmaAltitude = 0.0f;    // Altitude standard deviation in meters
    Eigen::Vector3f accStdDev = Eigen::Vector3f::Zero(); // Standard deviation for acceleration x,y,z in IMU sensor frame [m/(s*s)]
    Eigen::Vector3f gyrStdDev = Eigen::Vector3f::Zero(); // Standard deviation for angular rate around x-axis,y-axis,z-axis in IMU sensor frame [rad/s]

    // Clear all fields
    void clear() {
        SystemFailure = false;
        AccelerometerSensorFailure = false;
        GyroscopeSensorFailure = false;
        MagnetometerSensorFailure = false;
        GNSSFailureSecondaryAntenna = false;
        GNSSFailurePrimaryAntenna = false;
        AccelerometerOverRange = false;
        GyroscopeOverRange = false;
        MagnetometerOverRange = false;
        MinimumTemperatureAlarm = false;
        MaximumTemperatureAlarm = false;
        GNSSAntennaConnectionBroken = false;
        DataOutputOverflowAlarm = false;

        // Filter Status (Bytes 8-9)
        OrientationFilterInitialised = false;
        NavigationFilterInitialised = false;
        HeadingInitialised = false;
        UTCTimeInitialised = false;
        GNSSFixStatus = 0;
        Event1 = false;
        Event2 = false;
        InternalGNSSEnabled = false;
        DualAntennaHeadingActive = false;
        VelocityHeadingEnabled = false;
        GNSSFixInterrupted = false;
        ExternalPositionActive = false;
        ExternalVelocityActive = false;
        ExternalHeadingActive = false;

        // Navigation Data
        timestamp = 0.0;
        latitude = 0.0;
        longitude = 0.0;
        altitude = 0.0;
        velocityNorth = 0.0f;
        velocityEast = 0.0f;
        velocityDown = 0.0f;
        accelX = 0.0f;
        accelY = 0.0f;
        accelZ = 0.0f;
        gForce = 0.0f;
        roll = 0.0f;
        pitch = 0.0f;
        yaw = 0.0f;
        orientation = Eigen::Quaternionf::Identity();
        angularVelocityX = 0.0f;
        angularVelocityY = 0.0f;
        angularVelocityZ = 0.0f;
        sigmaLatitude = 0.0f;
        sigmaLongitude = 0.0f;
        sigmaAltitude = 0.0f;
        accStdDev = Eigen::Vector3f::Zero();
        gyrStdDev = Eigen::Vector3f::Zero();
    }

    [[nodiscard]] ImuData toImuData() const {
        ImuData imudata;
        imudata.acc = Eigen::Vector3f(this->accelX, this->accelY, this->accelZ);
        imudata.gyr = Eigen::Vector3f(this->angularVelocityX, this->angularVelocityY, this->angularVelocityZ);
        imudata.accStdDev = this->accStdDev;
        imudata.gyrStdDev = this->gyrStdDev;
        imudata.timestamp = this->timestamp;
        return imudata;
    }

    [[nodiscard]] PositionData toPositionData() const {
        PositionData positiondata;
        positiondata.pose = Eigen::Vector3d(this->latitude, this->longitude, this->altitude);
        positiondata.euler = Eigen::Vector3f(this->roll, this->pitch, this->yaw);
        positiondata.orientation = this->orientation;
        positiondata.poseStdDev = Eigen::Vector3f(this->sigmaLatitude, this->sigmaLongitude, this->sigmaAltitude);
        positiondata.timestamp = this->timestamp;
        return positiondata;
    }

    [[nodiscard]] CompFrame linearInterpolate(const CompFrame& a, const CompFrame& b, double t) const {
        CompFrame result;

        // Clamp t to [0, 1] to avoid extrapolation
        t = std::max(0.0, std::min(1.0, t));

        // Numeric fields: Linear interpolation
        result.timestamp = a.timestamp + t * (b.timestamp - a.timestamp);
        result.latitude = a.latitude + t * (b.latitude - a.latitude);
        result.longitude = a.longitude + t * (b.longitude - a.longitude);
        result.altitude = a.altitude + t * (b.altitude - a.altitude);
        result.velocityNorth = a.velocityNorth + t * (b.velocityNorth - a.velocityNorth);
        result.velocityEast = a.velocityEast + t * (b.velocityEast - a.velocityEast);
        result.velocityDown = a.velocityDown + t * (b.velocityDown - a.velocityDown);
        result.accelX = a.accelX + t * (b.accelX - a.accelX);
        result.accelY = a.accelY + t * (b.accelY - a.accelY);
        result.accelZ = a.accelZ + t * (b.accelZ - a.accelZ);
        result.gForce = a.gForce + t * (b.gForce - a.gForce);
        result.roll = a.roll + t * (b.roll - a.roll);
        result.pitch = a.pitch + t * (b.pitch - a.pitch);
        result.yaw = a.yaw + t * (b.yaw - a.yaw);

        // Quaternion: Spherical linear interpolation
        result.orientation = a.orientation.slerp(t, b.orientation);

        result.angularVelocityX = a.angularVelocityX + t * (b.angularVelocityX - a.angularVelocityX);
        result.angularVelocityY = a.angularVelocityY + t * (b.angularVelocityY - a.angularVelocityY);
        result.angularVelocityZ = a.angularVelocityZ + t * (b.angularVelocityZ - a.angularVelocityZ);
        result.sigmaLatitude = a.sigmaLatitude + t * (b.sigmaLatitude - a.sigmaLatitude);
        result.sigmaLongitude = a.sigmaLongitude + t * (b.sigmaLongitude - a.sigmaLongitude);
        result.sigmaAltitude = a.sigmaAltitude + t * (b.sigmaAltitude - a.sigmaAltitude);

        // Eigen vectors: Linear interpolation
        result.accStdDev = a.accStdDev + t * (b.accStdDev - a.accStdDev);
        result.gyrStdDev = a.gyrStdDev + t * (b.gyrStdDev - a.gyrStdDev);

        // System Status: Set to true if either frame indicates an issue (conservative)
        result.SystemFailure = a.SystemFailure || b.SystemFailure;
        result.AccelerometerSensorFailure = a.AccelerometerSensorFailure || b.AccelerometerSensorFailure;
        result.GyroscopeSensorFailure = a.GyroscopeSensorFailure || b.GyroscopeSensorFailure;
        result.MagnetometerSensorFailure = a.MagnetometerSensorFailure || b.MagnetometerSensorFailure;
        result.GNSSFailureSecondaryAntenna = a.GNSSFailureSecondaryAntenna || b.GNSSFailureSecondaryAntenna;
        result.GNSSFailurePrimaryAntenna = a.GNSSFailurePrimaryAntenna || b.GNSSFailurePrimaryAntenna;
        result.AccelerometerOverRange = a.AccelerometerOverRange || b.AccelerometerOverRange;
        result.GyroscopeOverRange = a.GyroscopeOverRange || b.GyroscopeOverRange;
        result.MagnetometerOverRange = a.MagnetometerOverRange || b.MagnetometerOverRange;
        result.MinimumTemperatureAlarm = a.MinimumTemperatureAlarm || b.MinimumTemperatureAlarm;
        result.MaximumTemperatureAlarm = a.MaximumTemperatureAlarm || b.MaximumTemperatureAlarm;
        result.GNSSAntennaConnectionBroken = a.GNSSAntennaConnectionBroken || b.GNSSAntennaConnectionBroken;
        result.DataOutputOverflowAlarm = a.DataOutputOverflowAlarm || b.DataOutputOverflowAlarm;

        // Filter Status: Set to true only if both frames agree (conservative for initialization)
        result.OrientationFilterInitialised = a.OrientationFilterInitialised && b.OrientationFilterInitialised;
        result.NavigationFilterInitialised = a.NavigationFilterInitialised && b.NavigationFilterInitialised;
        result.HeadingInitialised = a.HeadingInitialised && b.HeadingInitialised;
        result.UTCTimeInitialised = a.UTCTimeInitialised && b.UTCTimeInitialised;

        // GNSSFixStatus: Use the value from the closer frame
        result.GNSSFixStatus = (t < 0.5) ? a.GNSSFixStatus : b.GNSSFixStatus;
        result.Event1 = a.Event1 || b.Event1;
        result.Event2 = a.Event2 || b.Event2;
        result.InternalGNSSEnabled = a.InternalGNSSEnabled && b.InternalGNSSEnabled;
        result.DualAntennaHeadingActive = a.DualAntennaHeadingActive && b.DualAntennaHeadingActive;
        result.VelocityHeadingEnabled = a.VelocityHeadingEnabled && b.VelocityHeadingEnabled;
        result.GNSSFixInterrupted = a.GNSSFixInterrupted || b.GNSSFixInterrupted;
        result.ExternalPositionActive = a.ExternalPositionActive && b.ExternalPositionActive;
        result.ExternalVelocityActive = a.ExternalVelocityActive && b.ExternalVelocityActive;
        result.ExternalHeadingActive = a.ExternalHeadingActive && b.ExternalHeadingActive;

        return result;
    }
};







