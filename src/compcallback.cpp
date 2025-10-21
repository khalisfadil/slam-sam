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

    InitializePool(poolsize_);
    active_frame_ = GetFrameFromPool();
}
// %             ... struct for parameter
CompCallback::CompCallback(const nlohmann::json& json_data) {
    ParseMetadata(json_data);
    InitializePool(poolsize_);
    active_frame_ = GetFrameFromPool();
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
        if (!imu_params.contains("staticBiasAccelerometer") || !imu_params["staticBiasAccelerometer"].is_array() || imu_params["staticBiasAccelerometer"].size() != 3) {
            throw std::runtime_error("'staticBiasAccelerometer' must be an array of 3 elements");
        }
        staticBiasAccelerometer_ = Eigen::Vector3d(
            imu_params["staticBiasAccelerometer"][0].get<double>(),
            imu_params["staticBiasAccelerometer"][1].get<double>(),
            imu_params["staticBiasAccelerometer"][2].get<double>()
        );

        // Parse angularRandomWalk (3-element array)
        if (!imu_params.contains("staticBiasGyroscope") || !imu_params["staticBiasGyroscope"].is_array() || imu_params["staticBiasGyroscope"].size() != 3) {
            throw std::runtime_error("'staticBiasGyroscope' must be an array of 3 elements");
        }
        staticBiasGyroscope_ = Eigen::Vector3d(
            imu_params["staticBiasGyroscope"][0].get<double>(),
            imu_params["staticBiasGyroscope"][1].get<double>(),
            imu_params["staticBiasGyroscope"][2].get<double>()
        );

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
        biasInstabilityAccelerometer_ = Eigen::Vector3d(
            imu_params["biasInstabilityAccelerometer"][0].get<double>(),
            imu_params["biasInstabilityAccelerometer"][1].get<double>(),
            imu_params["biasInstabilityAccelerometer"][2].get<double>()
        );

        // Parse biasInstabilityGyroscope (3-element array)
        if (!imu_params.contains("biasInstabilityGyroscope") || !imu_params["biasInstabilityGyroscope"].is_array() || imu_params["biasInstabilityGyroscope"].size() != 3) {
            throw std::runtime_error("'biasInstabilityGyroscope' must be an array of 3 elements");
        }
        biasInstabilityGyroscope_ = Eigen::Vector3d(
            imu_params["biasInstabilityGyroscope"][0].get<double>(),
            imu_params["biasInstabilityGyroscope"][1].get<double>(),
            imu_params["biasInstabilityGyroscope"][2].get<double>()
        );

        // Parse biasInstabilityAccelerometer (3-element array)
        if (!imu_params.contains("biasRandomWalkAccelerometer") || !imu_params["biasRandomWalkAccelerometer"].is_array() || imu_params["biasRandomWalkAccelerometer"].size() != 3) {
            throw std::runtime_error("'biasRandomWalkAccelerometer' must be an array of 3 elements");
        }
        biasRandomWalkAccelerometer_ = Eigen::Vector3d(
            imu_params["biasRandomWalkAccelerometer"][0].get<double>(),
            imu_params["biasRandomWalkAccelerometer"][1].get<double>(),
            imu_params["biasRandomWalkAccelerometer"][2].get<double>()
        );

        // Parse biasInstabilityGyroscope (3-element array)
        if (!imu_params.contains("biasRandomWalkGyroscope") || !imu_params["biasRandomWalkGyroscope"].is_array() || imu_params["biasRandomWalkGyroscope"].size() != 3) {
            throw std::runtime_error("'biasRandomWalkGyroscope' must be an array of 3 elements");
        }
        biasRandomWalkGyroscope_ = Eigen::Vector3d(
            imu_params["biasRandomWalkGyroscope"][0].get<double>(),
            imu_params["biasRandomWalkGyroscope"][1].get<double>(),
            imu_params["biasRandomWalkGyroscope"][2].get<double>()
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
        if (!imu_params.contains("poolSize") || !imu_params["poolSize"].is_number()) {
            throw std::runtime_error("Missing or invalid 'poolSize'");
        }
        poolsize_ = imu_params["poolsize"].get<size_t>();

    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("JSON parsing error in metadata: " + std::string(e.what()));
    }
}
// %            ... sensor fusion constructor
std::unique_ptr<CompFrame> CompCallback::DecodePacket(const std::vector<uint8_t>& packet) {

    std::unique_ptr<CompFrame> completed_frame = nullptr;

    static constexpr size_t header_size_ = 5;
     // 5-byte header + 100-byte data
    static constexpr uint8_t packet_id_20 = 0x14;   // Packet ID is 20 (0x14)
    static constexpr size_t data_size_20 = 100;    // Packet data length is 100 bytes
    static constexpr size_t expected_size_20 = data_size_20 + header_size_;

    static constexpr uint8_t packet_id_25 = 0x19;   // Packet ID is 25 (0x19)
    static constexpr size_t data_size_25 = 12; 
    static constexpr size_t expected_size_25 = data_size_25 + header_size_;
    
    static constexpr uint8_t packet_id_26 = 0x1A;   // Packet ID is 26 (0x1A)
    static constexpr size_t data_size_26 = 12; 
    static constexpr size_t expected_size_26 = data_size_26 + header_size_;

    static constexpr uint8_t packet_id_28 = 0x1C;   // Packet ID is 28 (0x1C)
    static constexpr size_t data_size_28 = 48;    // Packet data length is 100 bytes
    static constexpr size_t expected_size_28 = data_size_28 + header_size_;
    
    static constexpr uint8_t packet_id_29 = 0x1D;   // Packet ID is 29 (0x1D)
    static constexpr size_t data_size_29 = 74; 
    static constexpr size_t expected_size_29 = data_size_29 + header_size_;

    // Validate Packet ID (at offset 1)
    uint8_t packet_id = packet[1];
    uint8_t packet_length = packet[2];

    // Use a switch statement for clear logic
    switch (packet_id) {
        case packet_id_20: {
            // ... validate size and decode packet 20 ...
            if (packet.size() != data_size_20 + header_size_ || packet_length != data_size_20) {
                return nullptr; // Invalid size or length for this packet ID
            }
            // System Status (Bytes 5-6, u16)
            uint16_t system_status_raw;
            std::memcpy(&system_status_raw, packet.data() + 5, sizeof(uint16_t));
            uint16_t system_status = le16toh(system_status_raw);
            active_frame_->SystemFailure_20 = (system_status & 0x0001) != 0;
            active_frame_->AccelerometerSensorFailure_20 = (system_status & 0x0002) != 0;
            active_frame_->GyroscopeSensorFailure_20 = (system_status & 0x0004) != 0;
            active_frame_->MagnetometerSensorFailure_20 = (system_status & 0x0008) != 0;
            active_frame_->GNSSFailureSecondaryAntenna_20 = (system_status & 0x0010) != 0;
            active_frame_->GNSSFailurePrimaryAntenna_20 = (system_status & 0x0020) != 0;
            active_frame_->AccelerometerOverRange_20 = (system_status & 0x0040) != 0;
            active_frame_->GyroscopeOverRange_20 = (system_status & 0x0080) != 0;
            active_frame_->MagnetometerOverRange_20 = (system_status & 0x0100) != 0;
            active_frame_->MinimumTemperatureAlarm_20 = (system_status & 0x0400) != 0;
            active_frame_->MaximumTemperatureAlarm_20 = (system_status & 0x0800) != 0;
            active_frame_->GNSSAntennaConnectionBroken_20 = (system_status & 0x4000) != 0;
            active_frame_->DataOutputOverflowAlarm_20 = (system_status & 0x8000) != 0;

            // Filter Status (Bytes 7-8, u16)
            uint16_t filter_status_raw;
            std::memcpy(&filter_status_raw, packet.data() + 7, sizeof(uint16_t));
            uint16_t filter_status = le16toh(filter_status_raw);
            active_frame_->OrientationFilterInitialised_20 = (filter_status & 0x0001) != 0;
            active_frame_->NavigationFilterInitialised_20 = (filter_status & 0x0002) != 0;
            active_frame_->HeadingInitialised_20 = (filter_status & 0x0004) != 0;
            active_frame_->UTCTimeInitialised_20 = (filter_status & 0x0008) != 0;
            active_frame_->GNSSFixStatus_20 = (filter_status >> 4) & 0x07;
            active_frame_->Event1_20 = (filter_status & 0x0080) != 0;
            active_frame_->Event2_20 = (filter_status & 0x0100) != 0;
            active_frame_->InternalGNSSEnabled_20 = (filter_status & 0x0200) != 0;
            active_frame_->DualAntennaHeadingActive_20 = (filter_status & 0x0400) != 0;
            active_frame_->VelocityHeadingEnabled_20 = (filter_status & 0x0800) != 0;
            active_frame_->GNSSFixInterrupted_20 = (filter_status & 0x1000) != 0;
            active_frame_->ExternalPositionActive_20 = (filter_status & 0x2000) != 0;
            active_frame_->ExternalVelocityActive_20 = (filter_status & 0x4000) != 0;
            active_frame_->ExternalHeadingActive_20 = (filter_status & 0x8000) != 0;

            // Unix Time (Bytes 9-16, u32 seconds + u32 microseconds)
            uint32_t seconds_raw, microseconds_raw;
            std::memcpy(&seconds_raw, packet.data() + 9, sizeof(uint32_t));
            std::memcpy(&microseconds_raw, packet.data() + 13, sizeof(uint32_t));
            uint32_t seconds = le32toh(seconds_raw);
            uint32_t microseconds = le32toh(microseconds_raw);
            if (microseconds > 999999) {
                return nullptr;
            }
            double timestamp = static_cast<double>(seconds) + static_cast<double>(microseconds) * 1e-6;
            active_frame_->timestamp_20 = std::fmod(timestamp, 86400.0); // Seconds since midnight UTC

            // Latitude, Longitude, Altitude (Bytes 17-40, fp64)
            std::memcpy(&active_frame_->latitude_20, packet.data() + 17, sizeof(double));
            std::memcpy(&active_frame_->longitude_20, packet.data() + 25, sizeof(double));
            std::memcpy(&active_frame_->altitude_20, packet.data() + 33, sizeof(double));

            // Velocity North, East, Down (Bytes 41-52, fp32)
            std::memcpy(&active_frame_->velocityNorth_20, packet.data() + 41, sizeof(float));
            std::memcpy(&active_frame_->velocityEast_20, packet.data() + 45, sizeof(float));
            std::memcpy(&active_frame_->velocityDown_20, packet.data() + 49, sizeof(float));

            // Body Acceleration X, Y, Z (Bytes 53-64, fp32, treat as sensor frame IMU data)
            std::memcpy(&active_frame_->accelX_20, packet.data() + 53, sizeof(float));
            std::memcpy(&active_frame_->accelY_20, packet.data() + 57, sizeof(float));
            std::memcpy(&active_frame_->accelZ_20, packet.data() + 61, sizeof(float));

            // G Force (Bytes 65-68, fp32)
            std::memcpy(&active_frame_->gForce_20, packet.data() + 65, sizeof(float));

            // Roll, Pitch, Yaw (Bytes 69-80, fp32)
            std::memcpy(&active_frame_->roll_20, packet.data() + 69, sizeof(float));
            std::memcpy(&active_frame_->pitch_20, packet.data() + 73, sizeof(float));
            std::memcpy(&active_frame_->yaw_20, packet.data() + 77, sizeof(float));

            // Angular Velocity X, Y, Z (Bytes 81-92, fp32, treat as sensor frame IMU data)
            std::memcpy(&active_frame_->angularVelocityX_20, packet.data() + 81, sizeof(float));
            std::memcpy(&active_frame_->angularVelocityY_20, packet.data() + 85, sizeof(float));
            std::memcpy(&active_frame_->angularVelocityZ_20, packet.data() + 89, sizeof(float));

            // Standard Deviations (Bytes 93-104, fp32)
            std::memcpy(&active_frame_->sigmaLatitude_20, packet.data() + 93, sizeof(float));
            std::memcpy(&active_frame_->sigmaLongitude_20, packet.data() + 97, sizeof(float));
            std::memcpy(&active_frame_->sigmaAltitude_20, packet.data() + 101, sizeof(float));

            // Convert Euler angles (ZYX convention) to quaternion using Eigen
            Eigen::AngleAxisf rollAngle(active_frame_->roll_20, Eigen::Vector3f::UnitX());
            Eigen::AngleAxisf pitchAngle(active_frame_->pitch_20, Eigen::Vector3f::UnitY());
            Eigen::AngleAxisf yawAngle(active_frame_->yaw_20, Eigen::Vector3f::UnitZ());

            Eigen::Quaternionf orientation = yawAngle * pitchAngle * rollAngle;
            active_frame_->qw_20 = orientation.w();
            active_frame_->qx_20 = orientation.x();
            active_frame_->qy_20 = orientation.y();
            active_frame_->qz_20 = orientation.z();
            
            active_frame_-> valid_20 = true;
            break;
        }

        case packet_id_25: {
            // ... validate size and decode packet 25 ...
            if (packet.size() != data_size_25 + header_size_ || packet_length != data_size_25) {
                return nullptr; // Invalid size or length for this packet ID
            }
            // Velocity Standard Deviations (Bytes 5-16, fp32)
            std::memcpy(&active_frame_->sigmaVelocityNorth_25, packet.data() + 5, sizeof(float));
            std::memcpy(&active_frame_->sigmaVelocityEast_25, packet.data() + 9, sizeof(float));
            std::memcpy(&active_frame_->sigmaVelocityDown_25, packet.data() + 13, sizeof(float));

            active_frame_-> valid_25 = true;
            break;
        }

        case packet_id_26: {
            // ... validate size and decode packet 26 ...
            if (packet.size() != data_size_26 + header_size_ || packet_length != data_size_26) {
                return nullptr; // Invalid size or length for this packet ID
            }
            // Orientation Standard Deviations (Bytes 5-16, fp32)
            std::memcpy(&active_frame_->sigmaRoll_26, packet.data() + 5, sizeof(float));
            std::memcpy(&active_frame_->sigmaPitch_26, packet.data() + 9, sizeof(float));
            std::memcpy(&active_frame_->sigmaYaw_26, packet.data() + 13, sizeof(float));

            active_frame_-> valid_26 = true;
            break;
        }

        case packet_id_28: {
            // ... validate size and decode packet 28 ...
            if (packet.size() != data_size_28 + header_size_ || packet_length != data_size_28) {
                return nullptr; // Invalid size or length for this packet ID
            }
            // IMU and Environmental Sensor Measurements (Bytes 5-52, fp32)
            std::memcpy(&active_frame_->accelX_28, packet.data() + 5, sizeof(float));
            std::memcpy(&active_frame_->accelY_28, packet.data() + 9, sizeof(float));
            std::memcpy(&active_frame_->accelZ_28, packet.data() + 13, sizeof(float));
            std::memcpy(&active_frame_->gyroX_28, packet.data() + 17, sizeof(float));
            std::memcpy(&active_frame_->gyroY_28, packet.data() + 21, sizeof(float));
            std::memcpy(&active_frame_->gyroZ_28, packet.data() + 25, sizeof(float));
            std::memcpy(&active_frame_->magX_28, packet.data() + 29, sizeof(float));
            std::memcpy(&active_frame_->magY_28, packet.data() + 33, sizeof(float));
            std::memcpy(&active_frame_->magZ_28, packet.data() + 37, sizeof(float));
            std::memcpy(&active_frame_->imuTemperature_28, packet.data() + 41, sizeof(float));
            std::memcpy(&active_frame_->pressure_28, packet.data() + 45, sizeof(float));
            std::memcpy(&active_frame_->pressureTemperature_28, packet.data() + 49, sizeof(float));

            active_frame_-> valid_28 = true;
            break;
        }

        case packet_id_29: {
            // ... validate size and decode packet 29 ...
            if (packet.size() != data_size_29 + header_size_ || packet_length != data_size_29) {
                return nullptr; // Invalid size or length for this packet ID
            }
            // (Bytes 5-12, u32 seconds + u32 microseconds)
            uint32_t seconds_raw, microseconds_raw;
            std::memcpy(&seconds_raw, packet.data() + 5, sizeof(uint32_t));
            std::memcpy(&microseconds_raw, packet.data() + 9, sizeof(uint32_t));
            uint32_t seconds = le32toh(seconds_raw);
            uint32_t microseconds = le32toh(microseconds_raw);
            if (microseconds > 999999) {
                return nullptr;
            }
            double timestamp = static_cast<double>(seconds) + static_cast<double>(microseconds) * 1e-6;
            active_frame_->timestamp_29 = std::fmod(timestamp, 86400.0);

            // Latitude, Longitude, Height (Bytes 13-36, fp64)
            std::memcpy(&active_frame_->latitude_29, packet.data() + 13, sizeof(double));
            std::memcpy(&active_frame_->longitude_29, packet.data() + 21, sizeof(double));
            std::memcpy(&active_frame_->altitude_29, packet.data() + 29, sizeof(double));

            // Velocity North, East, Down (Bytes 37-48, fp32)
            std::memcpy(&active_frame_->velocityNorth_29, packet.data() + 37, sizeof(float));
            std::memcpy(&active_frame_->velocityEast_29, packet.data() + 41, sizeof(float));
            std::memcpy(&active_frame_->velocityDown_29, packet.data() + 45, sizeof(float));

            // Standard Deviations (Bytes 49-60, fp32)
            std::memcpy(&active_frame_->sigmaLatitude_29, packet.data() + 49, sizeof(float));
            std::memcpy(&active_frame_->sigmaLongitude_29, packet.data() + 53, sizeof(float));
            std::memcpy(&active_frame_->sigmaAltitude_29, packet.data() + 57, sizeof(float));

            // Tilt and Heading (Bytes 61-68, fp32)
            std::memcpy(&active_frame_->tilt_29, packet.data() + 61, sizeof(float));
            std::memcpy(&active_frame_->heading_29, packet.data() + 65, sizeof(float));

            // Tilt and Heading Standard Deviations (Bytes 69-76, fp32)
            std::memcpy(&active_frame_->sigmaTilt_29, packet.data() + 69, sizeof(float));
            std::memcpy(&active_frame_->sigmaHeading_29, packet.data() + 73, sizeof(float));

            // Status (Bytes 77-78, u16)
            uint16_t status_raw;
            std::memcpy(&status_raw, packet.data() + 77, sizeof(uint16_t));
            uint16_t status = le16toh(status_raw);
            active_frame_->GNSSFixStatus_29 = status & 0x07; // Bits 0-2
            active_frame_->dopplerVelocityValid_29 = (status & 0x08) != 0; // Bit 3
            active_frame_->timeValid_29 = (status & 0x10) != 0; // Bit 4
            active_frame_->externalGNSS_29 = (status & 0x20) != 0; // Bit 5
            active_frame_->tiltValid_29 = (status & 0x40) != 0; // Bit 6

            active_frame_-> valid_29 = true;
            break;
        }
    }
    if (active_frame_->isValid()) {
        completed_frame = std::move(active_frame_);
        active_frame_ = GetFrameFromPool();
    }
    return completed_frame;
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
// %             ... WGS84 Gravity function
void CompCallback::InitializePool(size_t pool_size) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    frame_pool_.clear();
    for (size_t i = 0; i < pool_size; ++i) {
        auto frame = std::make_unique<CompFrame>();
        frame_pool_.push_back(std::move(frame));
    }
}
// %             ... WGS84 Gravity function
std::unique_ptr<CompFrame> CompCallback::GetFrameFromPool() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (!frame_pool_.empty()) {
        std::unique_ptr<CompFrame> frame = std::move(frame_pool_.front());
        frame_pool_.pop_front();
        return frame;
    }
    auto frame = std::make_unique<CompFrame>();
    return frame;
}
// %             ... WGS84 Gravity function
void CompCallback::ReturnFrameToPool(std::unique_ptr<CompFrame> frame) {
    if (frame) {
        frame->clear(); // Reset the frame for reuse
        std::lock_guard<std::mutex> lock(pool_mutex_);
        frame_pool_.push_back(std::move(frame));
    }
}


