#include <compcallback.hpp> 

using json = nlohmann::json;

CompCallback::CompCallback(const std::string& json_path){
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + json_path);
    }
    json json_data;
    try {
        file >> json_data;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("JSON parse error in " + json_path + ": " + e.what());
    }
    ParseMetadata(json_data);
}
// %             ... struct for parameter
CompCallback::CompCallback(const nlohmann::json& json_data) {
    ParseMetadata(json_data);
}
// %            ... sensor fusion constructor
void CompCallback::ParseMetadata(const nlohmann::json& json_data) {
    if (!json_data.is_object()) {
        throw std::runtime_error("JSON data must be an object");
    }

    // Store the JSON data
    metadata_ = json_data;

    try {
        // Check for imu_parameter object
        if (!metadata_.contains("imu_parameter") || !metadata_["imu_parameter"].is_object()) {
            throw std::runtime_error("Missing or invalid 'imu_parameter' object");
        }
        const auto& imu_params = metadata_["imu_parameter"];

        // Parse updateRateHz
        if (!imu_params.contains("updateRateHz") || !imu_params["updateRateHz"].is_number()) {
            throw std::runtime_error("Missing or invalid 'updateRateHz'");
        }
        updateRate_ = imu_params["updateRateHz"].get<double>();

        // Parse velocityRandomWalk (3-element array)
        if (!imu_params.contains("velocityRandomWalk") || !imu_params["velocityRandomWalk"].is_array() || imu_params["velocityRandomWalk"].size() != 3) {
            throw std::runtime_error("'velocityRandomWalk' must be an array of 3 elements");
        }
        velocityRandomWalk_ = Eigen::Vector3d(
            imu_params["velocityRandomWalk"][0].get<double>(),
            imu_params["velocityRandomWalk"][1].get<double>(),
            imu_params["velocityRandomWalk"][2].get<double>()
        );

        // Parse angularRandomWalk (3-element array)
        if (!imu_params.contains("angularRandomWalk") || !imu_params["angularRandomWalk"].is_array() || imu_params["angularRandomWalk"].size() != 3) {
            throw std::runtime_error("'angularRandomWalk' must be an array of 3 elements");
        }
        angularRandomWalk_ = Eigen::Vector3d(
            imu_params["angularRandomWalk"][0].get<double>(),
            imu_params["angularRandomWalk"][1].get<double>(),
            imu_params["angularRandomWalk"][2].get<double>()
        );

        // Parse biasInstabilityAccelerometer (3-element array)
        if (!imu_params.contains("biasInstabilityAccelerometer") || !imu_params["biasInstabilityAccelerometer"].is_array() || imu_params["biasInstabilityAccelerometer"].size() != 3) {
            throw std::runtime_error("'biasInstabilityAccelerometer' must be an array of 3 elements");
        }
        biasAccelerometer_ = Eigen::Vector3d(
            imu_params["biasInstabilityAccelerometer"][0].get<double>(),
            imu_params["biasInstabilityAccelerometer"][1].get<double>(),
            imu_params["biasInstabilityAccelerometer"][2].get<double>()
        );

        // Parse biasInstabilityGyroscope (3-element array)
        if (!imu_params.contains("biasInstabilityGyroscope") || !imu_params["biasInstabilityGyroscope"].is_array() || imu_params["biasInstabilityGyroscope"].size() != 3) {
            throw std::runtime_error("'biasInstabilityGyroscope' must be an array of 3 elements");
        }
        biasGyroscope_ = Eigen::Vector3d(
            imu_params["biasInstabilityGyroscope"][0].get<double>(),
            imu_params["biasInstabilityGyroscope"][1].get<double>(),
            imu_params["biasInstabilityGyroscope"][2].get<double>()
        );

        // Parse position (3-element array)
        if (!imu_params.contains("tb2s") || !imu_params["tb2s"].is_array() || imu_params["tb2s"].size() != 3) {
            throw std::runtime_error("'position' must be an array of 3 elements");
        }
        body_to_imu_translation_ = Eigen::Vector3d(
            imu_params["tb2s"][0].get<double>(),
            imu_params["tb2s"][1].get<double>(),
            imu_params["tb2s"][2].get<double>()
        );

        // Parse rotationMatrix (9-element array, row-major)
        if (!imu_params.contains("Cb2s") || !imu_params["Cb2s"].is_array() || imu_params["Cb2s"].size() != 9) {
            throw std::runtime_error("'Cb2s' must be an array of 9 elements");
        }
        body_to_imu_rotation_ = Eigen::Matrix3d();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                body_to_imu_rotation_(i, j) = imu_params["Cb2s"][i * 3 + j].get<double>();
            }
        }

    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("JSON parsing error in metadata: " + std::string(e.what()));
    }
}
// %            ... sensor fusion constructor
void CompCallback::Decode(const std::vector<uint8_t>& packet, CompFrame& frame) {
    // Define static expected size
    static constexpr size_t expected_size_ = 105; // 5-byte header + 100-byte data
    static constexpr uint8_t packet_id_ = 0x14;   // Packet ID is 20 (0x14)
    static constexpr uint8_t data_size_ = 100;    // Packet data length is 100 bytes

    // Validate packet size
    if (packet.size() != expected_size_) {
        return;
    }

    // Validate Packet ID (at offset 1)
    uint8_t packet_id = packet[1];
    if (packet_id != packet_id_) {
        return;
    }

    // Validate Packet Length (at offset 2)
    uint8_t packet_length = packet[2];
    if (packet_length != data_size_) {
        return;
    }

    // Clear the output frame
    frame.clear();

    // System Status (Bytes 5-6, u16)
    uint16_t system_status_raw;
    std::memcpy(&system_status_raw, packet.data() + 5, sizeof(uint16_t));
    uint16_t system_status = le16toh(system_status_raw);
    frame.SystemFailure = (system_status & 0x0001) != 0;
    frame.AccelerometerSensorFailure = (system_status & 0x0002) != 0;
    frame.GyroscopeSensorFailure = (system_status & 0x0004) != 0;
    frame.MagnetometerSensorFailure = (system_status & 0x0008) != 0;
    frame.GNSSFailureSecondaryAntenna = (system_status & 0x0010) != 0;
    frame.GNSSFailurePrimaryAntenna = (system_status & 0x0020) != 0;
    frame.AccelerometerOverRange = (system_status & 0x0040) != 0;
    frame.GyroscopeOverRange = (system_status & 0x0080) != 0;
    frame.MagnetometerOverRange = (system_status & 0x0100) != 0;
    frame.MinimumTemperatureAlarm = (system_status & 0x0400) != 0;
    frame.MaximumTemperatureAlarm = (system_status & 0x0800) != 0;
    frame.GNSSAntennaConnectionBroken = (system_status & 0x4000) != 0;
    frame.DataOutputOverflowAlarm = (system_status & 0x8000) != 0;

    // Filter Status (Bytes 7-8, u16)
    uint16_t filter_status_raw;
    std::memcpy(&filter_status_raw, packet.data() + 7, sizeof(uint16_t));
    uint16_t filter_status = le16toh(filter_status_raw);
    frame.OrientationFilterInitialised = (filter_status & 0x0001) != 0;
    frame.NavigationFilterInitialised = (filter_status & 0x0002) != 0;
    frame.HeadingInitialised = (filter_status & 0x0004) != 0;
    frame.UTCTimeInitialised = (filter_status & 0x0008) != 0;
    frame.GNSSFixStatus = (filter_status >> 4) & 0x07;
    frame.Event1 = (filter_status & 0x0080) != 0;
    frame.Event2 = (filter_status & 0x0100) != 0;
    frame.InternalGNSSEnabled = (filter_status & 0x0200) != 0;
    frame.DualAntennaHeadingActive = (filter_status & 0x0400) != 0;
    frame.VelocityHeadingEnabled = (filter_status & 0x0800) != 0;
    frame.GNSSFixInterrupted = (filter_status & 0x1000) != 0;
    frame.ExternalPositionActive = (filter_status & 0x2000) != 0;
    frame.ExternalVelocityActive = (filter_status & 0x4000) != 0;
    frame.ExternalHeadingActive = (filter_status & 0x8000) != 0;

    // Unix Time (Bytes 9-16, u32 seconds + u32 microseconds)
    uint32_t seconds_raw, microseconds_raw;
    std::memcpy(&seconds_raw, packet.data() + 9, sizeof(uint32_t));
    std::memcpy(&microseconds_raw, packet.data() + 13, sizeof(uint32_t));
    uint32_t seconds = le32toh(seconds_raw);
    uint32_t microseconds = le32toh(microseconds_raw);
    if (microseconds > 999999) {
        return;
    }
    double timestamp = static_cast<double>(seconds) + static_cast<double>(microseconds) * 1e-6;
    frame.timestamp = std::fmod(timestamp, 86400.0); // Seconds since midnight UTC

    // Latitude, Longitude, Altitude (Bytes 17-40, fp64)
    std::memcpy(&frame.latitude, packet.data() + 17, sizeof(double));
    std::memcpy(&frame.longitude, packet.data() + 25, sizeof(double));
    std::memcpy(&frame.altitude, packet.data() + 33, sizeof(double));

    // Velocity North, East, Down (Bytes 41-52, fp32)
    std::memcpy(&frame.velocityNorth, packet.data() + 41, sizeof(float));
    std::memcpy(&frame.velocityEast, packet.data() + 45, sizeof(float));
    std::memcpy(&frame.velocityDown, packet.data() + 49, sizeof(float));

    // Body Acceleration X, Y, Z (Bytes 53-64, fp32, treat as sensor frame IMU data)
    float accelX_raw, accelY_raw, accelZ_raw;
    std::memcpy(&accelX_raw, packet.data() + 53, sizeof(float));
    std::memcpy(&accelY_raw, packet.data() + 57, sizeof(float));
    std::memcpy(&accelZ_raw, packet.data() + 61, sizeof(float));

    // G Force (Bytes 65-68, fp32)
    std::memcpy(&frame.gForce, packet.data() + 65, sizeof(float));

    // Roll, Pitch, Yaw (Bytes 69-80, fp32)
    float roll, pitch, yaw;
    std::memcpy(&roll, packet.data() + 69, sizeof(float));
    std::memcpy(&pitch, packet.data() + 73, sizeof(float));
    std::memcpy(&yaw, packet.data() + 77, sizeof(float));

    // Angular Velocity X, Y, Z (Bytes 81-92, fp32, treat as sensor frame IMU data)
    float angularVelocityX_raw, angularVelocityY_raw, angularVelocityZ_raw;
    std::memcpy(&angularVelocityX_raw, packet.data() + 81, sizeof(float));
    std::memcpy(&angularVelocityY_raw, packet.data() + 85, sizeof(float));
    std::memcpy(&angularVelocityZ_raw, packet.data() + 89, sizeof(float));

    // Standard Deviations (Bytes 93-104, fp32)
    std::memcpy(&frame.sigmaLatitude, packet.data() + 93, sizeof(float));
    std::memcpy(&frame.sigmaLongitude, packet.data() + 97, sizeof(float));
    std::memcpy(&frame.sigmaAltitude, packet.data() + 101, sizeof(float));

    // Process IMU data: subtract biases
    Eigen::Vector3d acc_sensor(accelX_raw, accelY_raw, accelZ_raw);
    // acc_sensor -= biasAccelerometer_; // Subtract accelerometer bias

    // Convert Euler angles (ZYX convention) to quaternion
    frame.orientation = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) * 
                    Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) * 
                    Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());

    Eigen::Vector3d acc_body = body_to_imu_rotation_.transpose() * acc_sensor;
    frame.accelX = static_cast<float>(acc_body(0));
    frame.accelY = static_cast<float>(acc_body(1));
    frame.accelZ = static_cast<float>(acc_body(2));

    Eigen::Vector3d ang_sensor(angularVelocityX_raw, angularVelocityY_raw, angularVelocityZ_raw);
    // ang_sensor -= biasGyroscope_; // Subtract gyroscope bias

    Eigen::Vector3d ang_body = body_to_imu_rotation_.transpose() * ang_sensor;
    frame.angularVelocityX = static_cast<float>(ang_body(0));
    frame.angularVelocityY = static_cast<float>(ang_body(1));
    frame.angularVelocityZ = static_cast<float>(ang_body(2));

    // Compute standard deviations in sensor frame
    double dt = 1.0 / updateRate_;
    double std_acc = velocityRandomWalk_(0) / std::sqrt(dt);  // e.g., 0.000333333 / sqrt(1/200) ≈ 0.004714 m/s²
    double std_gyro = angularRandomWalk_(0) / std::sqrt(dt);  // e.g., 0.000023271 / sqrt(1/200) ≈ 0.000329 rad/s
    frame.accStdDev = (Eigen::Vector3d(std_acc, std_acc, std_acc)).cast<float>();
    frame.gyrStdDev = (Eigen::Vector3d(std_gyro, std_gyro, std_gyro)).cast<float>();

    // std::cout<<"raw imu-z: "<<accelZ_raw<<", update accel-z: "<< frame.accelZ<<  std::endl;
}
// %             ... WGS84 Gravity function
double CompCallback::GravityWGS84(double latitude, double longitude, double altitude) {
    double sinphi = std::sin(latitude);
    double cosphi = std::cos(latitude);
    double sinlambda = std::sin(longitude);
    double coslambda = std::cos(longitude);
    double sin2phi = sinphi * sinphi;
    double N = a / std::sqrt(1.0 - e2 * sin2phi);
    double x_rec = (N + altitude) * cosphi * coslambda;
    double y_rec = (N + altitude) * cosphi * sinlambda;
    double z_rec = (b_over_a * b_over_a * N + altitude) * sinphi;
    double D = x_rec * x_rec + y_rec * y_rec + z_rec * z_rec - E2;
    double u2 = 0.5 * D * (1.0 + std::sqrt(1.0 + 4.0 * E2 * z_rec * z_rec / (D * D)));
    double u2E2 = u2 + E2;
    double u = std::sqrt(u2);
    double beta = std::atan2(z_rec * std::sqrt(u2E2), u * std::sqrt(x_rec * x_rec + y_rec * y_rec));
    double sinbeta = std::sin(beta);
    double cosbeta = std::cos(beta);
    double sin2beta = sinbeta * sinbeta;
    double cos2beta = cosbeta * cosbeta;
    double w = std::sqrt((u2 + E2 * sin2beta) / u2E2);
    double q = 0.5 * ((1.0 + 3.0 * u2 / E2) * std::atan(E / u) - 3.0 * u / E);
    double qo = 0.5 * ((1.0 + 3.0 * b * b / E2) * std::atan(E / b) - 3.0 * b / E);
    double q_prime = 3.0 * ((1.0 + u2 / E2) * (1.0 - (u / E) * std::atan(E / u))) - 1.0;
    double cf_u = u * cos2beta * omega * omega / w;
    double cf_beta = std::sqrt(u2E2) * cosbeta * sinbeta * omega * omega / w;
    double gamma_u = -(GM / u2E2 + omega * omega * a * a * E * q_prime * (0.5 * sin2beta - 1.0 / 6.0) / (u2E2 * qo)) / w + cf_u;
    double gamma_beta = omega * omega * a * a * q * sinbeta * cosbeta / (std::sqrt(u2E2) * w * qo) - cf_beta;
    return std::sqrt(gamma_u * gamma_u + gamma_beta * gamma_beta);
}


