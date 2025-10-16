#include <fstream>
#include <stdexcept>
#include <cmath>
#include <cstring> 
#include <iostream> 
#include <algorithm> 

#include <lidarcallback.hpp>

#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <endian.h> 

using json = nlohmann::json;

// %            ... lidarcallback constructor
LidarCallback::LidarCallback(const std::string& json_meta_path, const std::string& json_param_path) {
    std::ifstream file_meta(json_meta_path);
    if (!file_meta.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + json_meta_path);
    }
    json json_meta_data;
    try {
        file_meta >> json_meta_data;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("JSON parse error in " + json_meta_path + ": " + e.what());
    }
    ParseMetadata(json_meta_data);
    std::ifstream file_param(json_param_path);
    if (!file_param.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + json_param_path);
    }
    json json_param_data;
    try {
        file_param >> json_param_data;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("JSON parse error in " + json_param_path + ": " + e.what());
    }
    ParseParamdata(json_param_data);
    if (channel_stride_ == 0) {
        throw std::runtime_error("Channel stride N must be positive");
    }
    if (channel_stride_ != 1 && channel_stride_ != 2 && channel_stride_ != 4 && channel_stride_ != 8 && channel_stride_ != 16) {
        throw std::runtime_error("Channel stride N must be one of 1, 2, 4, 8, or 16, got " + std::to_string(channel_stride_));
    }
    if (channel_stride_ > static_cast<uint16_t>(pixels_per_column_)) {
        throw std::runtime_error("Channel stride N (" + std::to_string(channel_stride_) + ") exceeds pixels_per_column_ (" + std::to_string(pixels_per_column_) + ")");
    }
    Initialize();
}
// %            ... sensor fusion constructor
LidarCallback::LidarCallback(const json& json_meta, const json& json_param) {
    ParseMetadata(json_meta);
    ParseParamdata(json_param);
    if (channel_stride_ == 0) {
        throw std::runtime_error("Channel stride N must be positive");
    }
    if (channel_stride_ != 1 && channel_stride_ != 2 && channel_stride_ != 4 && channel_stride_ != 8 && channel_stride_ != 16) {
        throw std::runtime_error("Channel stride N must be one of 1, 2, 4, 8, or 16, got " + std::to_string(channel_stride_));
    }
    if (channel_stride_ > static_cast<uint16_t>(pixels_per_column_)) {
        throw std::runtime_error("Channel stride N (" + std::to_string(channel_stride_) + ") exceeds pixels_per_column_ (" + std::to_string(pixels_per_column_) + ")");
    }
    Initialize();
}
// %            ... sensor fusion constructor
void LidarCallback::ParseMetadata(const json& json_meta_data) {
    if (!json_meta_data.is_object()) {
        throw std::runtime_error("JSON data must be an object");
    }
    // Store the provided JSON data; subsequent accesses use the metadata_ member.
    metadata_ = json_meta_data;
    try {
        if (!metadata_.contains("lidar_data_format") || !metadata_["lidar_data_format"].is_object()) {
            throw std::runtime_error("Missing or invalid 'lidar_data_format' object");
        }
        if (!metadata_.contains("config_params") || !metadata_["config_params"].is_object()) {
            throw std::runtime_error("Missing or invalid 'config_params' object");
        }
        if (!metadata_.contains("beam_intrinsics") || !metadata_["beam_intrinsics"].is_object()) {
            throw std::runtime_error("Missing or invalid 'beam_intrinsics' object");
        }
        if (!metadata_.contains("lidar_intrinsics") || !metadata_["lidar_intrinsics"].is_object() ||
            !metadata_["lidar_intrinsics"].contains("lidar_to_sensor_transform")) {
            throw std::runtime_error("Missing or invalid 'lidar_intrinsics.lidar_to_sensor_transform'");
        }
        columns_per_frame_ = metadata_["lidar_data_format"]["columns_per_frame"].get<int>();
        pixels_per_column_ = metadata_["lidar_data_format"]["pixels_per_column"].get<int>();
        columns_per_packet_ = metadata_["config_params"]["columns_per_packet"].get<int>();
        udp_profile_lidar_ = metadata_["config_params"]["udp_profile_lidar"].get<std::string>();
        const auto& beam_intrinsics = metadata_["beam_intrinsics"];
        lidar_origin_to_beam_origin_mm_ = beam_intrinsics["lidar_origin_to_beam_origin_mm"].get<float>();
        const auto& pixel_shift_by_row = metadata_["lidar_data_format"]["pixel_shift_by_row"];
        if (!pixel_shift_by_row.is_array() || pixel_shift_by_row.size() != static_cast<size_t>(pixels_per_column_)) {
            throw std::runtime_error("'pixel_shift_by_row' must be an array of " + std::to_string(pixels_per_column_) + " elements");
        }
        pixel_shifts_.resize(pixels_per_column_);
        for (int i = 0; i < pixels_per_column_; ++i) {
            pixel_shifts_[i] = pixel_shift_by_row[i].get<int>();
        }
        const auto& lidar_transform_json = metadata_["lidar_intrinsics"]["lidar_to_sensor_transform"];
        if (!lidar_transform_json.is_array() || lidar_transform_json.size() != 16) {
            throw std::runtime_error("'lidar_to_sensor_transform' must be an array of 16 elements");
        }
        // Eigen::Matrix4d raw_transform = Eigen::Matrix4d::Identity(); // Not strictly needed if lidar_to_sensor_transform_ is filled directly
        lidar_to_sensor_transform_ = Eigen::Matrix4d::Identity();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double value = lidar_transform_json[i * 4 + j].get<double>();
                // raw_transform(i, j) = value; // if needed for other purposes
                // Scale translation components (last column, rows 0-2) from mm to m
                if (j == 3 && i < 3) {
                    lidar_to_sensor_transform_(i, j) = value * 0.001;
                } else {
                    lidar_to_sensor_transform_(i, j) = value;
                }
            }
        }
    } catch (const json::exception& e) {
        throw std::runtime_error("JSON parsing error in metadata: " + std::string(e.what()));
    }
}
// %            ... initialize
void LidarCallback::ParseParamdata(const json& json_param_data) {
    if (!json_param_data.is_object()) {
        throw std::runtime_error("JSON data must be an object");
    }
    // Store the provided JSON data; subsequent accesses use the parameter_ member.
    parameter_ = json_param_data;
    try {
        if (!parameter_.contains("lidar_parameter") || !parameter_["lidar_parameter"].is_object()) {
            throw std::runtime_error("Missing or invalid 'lidar_parameter' object");
        }
        const auto& lidar_param = parameter_["lidar_parameter"];
        // Parse position (3-element array)
        if (lidar_param.contains("tb2s") || lidar_param["tb2s"].is_array() || lidar_param["tb2s"].size() == 3) {
            body_to_lidar_translation_ = Eigen::Vector3d(
                lidar_param["tb2s"][0].get<double>(),
                lidar_param["tb2s"][1].get<double>(),
                lidar_param["tb2s"][2].get<double>()
            );
        }
        if (lidar_param.contains("Cb2s") || lidar_param["Cb2s"].is_array() || lidar_param["Cb2s"].size() == 9) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    body_to_lidar_rotation_(i, j) = lidar_param["Cb2s"][i * 3 + j].get<double>();
                }
            }
        }
        if (lidar_param.contains("channelStride")) {
            channel_stride_ = lidar_param["channelStride"].get<uint16_t>();
        }
        if (lidar_param.contains("zAxisFilter")) {
            zfiltermin_ = lidar_param["zAxisFilter"][0].get<float>();
            zfiltermax_ = lidar_param["zAxisFilter"][1].get<float>();
        }
        if (lidar_param.contains("reflectionThreshold")) {
            reflectivity_threshold_ = lidar_param["reflectionThreshold"].get<float>();
        }
        if (lidar_param.contains("rangeFilter")) {
            rfiltermin_ = lidar_param["rangeFilter"][0].get<float>();
            rfiltermax_ = lidar_param["rangeFilter"][1].get<float>();
        }
        if (lidar_param.contains("vehicleFilterBox") && lidar_param["vehicleFilterBox"].is_object()) {
            const auto& box_param = lidar_param["vehicleFilterBox"];
            if (box_param.contains("center") && box_param["center"].is_array() && box_param["center"].size() == 3 &&
                box_param.contains("dimensions") && box_param["dimensions"].is_array() && box_param["dimensions"].size() == 3) {
                
                vehicle_box_center_ = Eigen::Vector3f(
                    box_param["center"][0].get<float>(),
                    box_param["center"][1].get<float>(),
                    box_param["center"][2].get<float>()
                );

                vehicle_box_dimensions_ = Eigen::Vector3f(
                    box_param["dimensions"][0].get<float>(),
                    box_param["dimensions"][1].get<float>(),
                    box_param["dimensions"][2].get<float>()
                );
            }
        }
    } catch (const json::exception& e) {
        throw std::runtime_error("JSON parsing error in metadata: " + std::string(e.what()));
    }
}
// %            ... initialize
void LidarCallback::Initialize() {

    if (udp_profile_lidar_ == "RNG19_RFL8_SIG16_NIR16") {
        PACKET_HEADER_BYTES = 32;
        PACKET_FOOTER_BYTES = 32;
        COLUMN_HEADER_BYTES = 12;
        CHANNEL_STRIDE_BYTES = 12;
        MEASUREMENT_BLOCK_STATUS_BYTES = 0;
    } else if (udp_profile_lidar_ == "LEGACY") {
        PACKET_HEADER_BYTES = 0;
        PACKET_FOOTER_BYTES = 0;
        COLUMN_HEADER_BYTES = 16;
        CHANNEL_STRIDE_BYTES = 12;
        MEASUREMENT_BLOCK_STATUS_BYTES = 4;
    } else {
        throw std::runtime_error("Unsupported udp_profile_lidar: " + udp_profile_lidar_);
    }

    block_size_ = COLUMN_HEADER_BYTES + (pixels_per_column_ * CHANNEL_STRIDE_BYTES) + MEASUREMENT_BLOCK_STATUS_BYTES;
    expected_size_ = PACKET_HEADER_BYTES + (columns_per_packet_ * block_size_) + PACKET_FOOTER_BYTES;

    const auto& beam_intrinsics = metadata_["beam_intrinsics"];
    const auto& azimuth_angles_json = beam_intrinsics["beam_azimuth_angles"];
    const auto& altitude_angles_json = beam_intrinsics["beam_altitude_angles"];

    if (!azimuth_angles_json.is_array() || azimuth_angles_json.size() != static_cast<size_t>(pixels_per_column_) ||
        !altitude_angles_json.is_array() || altitude_angles_json.size() != static_cast<size_t>(pixels_per_column_)) {
        throw std::runtime_error("Beam azimuth/altitude angles missing or wrong size in JSON.");
    }

    // Calculate number of channels in subset
    subset_channels_ = (pixels_per_column_ + channel_stride_ - 1) / channel_stride_; // Ceiling division

    // Original lookup tables
    sin_beam_azimuths_.resize(pixels_per_column_);
    cos_beam_azimuths_.resize(pixels_per_column_);
    sin_beam_altitudes_.resize(pixels_per_column_);
    cos_beam_altitudes_.resize(pixels_per_column_);
    r_max_.resize(pixels_per_column_, rfiltermax_);
    r_min_.resize(pixels_per_column_, rfiltermin_);
    x_1_.assign(columns_per_frame_, std::vector<float, Eigen::aligned_allocator<float>>(pixels_per_column_));
    y_1_.assign(columns_per_frame_, std::vector<float, Eigen::aligned_allocator<float>>(pixels_per_column_));
    z_1_.assign(columns_per_frame_, std::vector<float, Eigen::aligned_allocator<float>>(pixels_per_column_));
    x_2_.resize(columns_per_frame_);
    y_2_.resize(columns_per_frame_);
    z_2_.resize(columns_per_frame_);
    pixel_shifts_.resize(pixels_per_column_);

    // Subset lookup tables
    sin_beam_azimuths_subset_.resize(subset_channels_);
    cos_beam_azimuths_subset_.resize(subset_channels_);
    sin_beam_altitudes_subset_.resize(subset_channels_);
    cos_beam_altitudes_subset_.resize(subset_channels_);
    r_max_subset_.resize(subset_channels_, rfiltermax_);
    r_min_subset_.resize(subset_channels_, rfiltermin_);
    x_1_subset_.assign(columns_per_frame_, std::vector<float, Eigen::aligned_allocator<float>>(subset_channels_));
    y_1_subset_.assign(columns_per_frame_, std::vector<float, Eigen::aligned_allocator<float>>(subset_channels_));
    z_1_subset_.assign(columns_per_frame_, std::vector<float, Eigen::aligned_allocator<float>>(subset_channels_));
    pixel_shifts_subset_.resize(subset_channels_);
    subset_c_ids_.resize(subset_channels_);

    for (int i = 0; i < pixels_per_column_; ++i) {
        float az_deg = azimuth_angles_json[i].get<float>();
        float alt_deg = altitude_angles_json[i].get<float>();
        float az_rad = az_deg * static_cast<float>(M_PI) / 180.0f;
        float alt_rad = alt_deg * static_cast<float>(M_PI) / 180.0f;
        sin_beam_azimuths_[i] = std::sin(az_rad);
        cos_beam_azimuths_[i] = std::cos(az_rad);
        sin_beam_altitudes_[i] = std::sin(alt_rad);
        cos_beam_altitudes_[i] = std::cos(alt_rad);
        pixel_shifts_[i] = metadata_["lidar_data_format"]["pixel_shift_by_row"][i].get<int>();
        if (i % channel_stride_ == 0) {
            size_t subset_idx = i / channel_stride_;
            sin_beam_azimuths_subset_[subset_idx] = sin_beam_azimuths_[i];
            cos_beam_azimuths_subset_[subset_idx] = cos_beam_azimuths_[i];
            sin_beam_altitudes_subset_[subset_idx] = sin_beam_altitudes_[i];
            cos_beam_altitudes_subset_[subset_idx] = cos_beam_altitudes_[i];
            r_max_subset_[subset_idx] = r_max_[i];
            r_min_subset_[subset_idx] = r_min_[i];
            pixel_shifts_subset_[subset_idx] = pixel_shifts_[i];
            subset_c_ids_[subset_idx] = i;
        }
    }

    Eigen::Matrix4d lidar_to_body_transform = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d body_to_lidar_transform = Eigen::Matrix4d::Identity();
    body_to_lidar_transform.block<3,3>(0,0) = body_to_lidar_rotation_;
    body_to_lidar_transform.block<3,1>(0,3) = body_to_lidar_translation_;// Chain if needed, assuming this is the full transform
    lidar_to_body_transform = body_to_lidar_transform.inverse();

    Eigen::Matrix3d lidar_to_body_rotation = lidar_to_body_transform.block<3,3>(0,0);

    for (int m_id = 0; m_id < columns_per_frame_; ++m_id) {
        float measurement_azimuth_rad = 2.0f * static_cast<float>(M_PI) * (1.0f - (static_cast<float>(m_id) / static_cast<float>(columns_per_frame_)));
        float cos_meas_az = std::cos(measurement_azimuth_rad);
        float sin_meas_az = std::sin(measurement_azimuth_rad);

        Eigen::Vector4d offset_lidar_frame(
            lidar_origin_to_beam_origin_mm_ * 0.001f * cos_meas_az,
            lidar_origin_to_beam_origin_mm_ * 0.001f * sin_meas_az,
            0.0,
            1.0);

        Eigen::Vector4d offset_transformed = lidar_to_body_transform * offset_lidar_frame;
        x_2_[m_id] = static_cast<float>(offset_transformed.x());
        y_2_[m_id] = static_cast<float>(offset_transformed.y());
        z_2_[m_id] = static_cast<float>(offset_transformed.z());

        for (int ch = 0; ch < pixels_per_column_; ++ch) {
            float beam_az_rad = azimuth_angles_json[ch].get<float>() * static_cast<float>(M_PI) / 180.0f;
            float total_az_rad = measurement_azimuth_rad + beam_az_rad;
            float cos_total_az = std::cos(total_az_rad);
            float sin_total_az = std::sin(total_az_rad);
            float cos_alt = cos_beam_altitudes_[ch];
            float sin_alt = sin_beam_altitudes_[ch];

            // --- FIX: Use 3x3 rotation for the 3D direction vector ---
            Eigen::Vector3d dir_lidar_frame(cos_alt * cos_total_az, cos_alt * sin_total_az, sin_alt);
            Eigen::Vector3d dir_transformed = lidar_to_body_rotation * dir_lidar_frame; // Correct transformation

            // Eigen::Vector4d dir_lidar_frame(cos_alt * cos_total_az, cos_alt * sin_total_az, sin_alt, 0.0);
            // Eigen::Vector4d dir_transformed = lidar_to_body_transform * dir_lidar_frame;
            x_1_[m_id][ch] = static_cast<float>(dir_transformed.x());
            y_1_[m_id][ch] = static_cast<float>(dir_transformed.y());
            z_1_[m_id][ch] = static_cast<float>(dir_transformed.z());

            if (ch % channel_stride_ == 0) {
                size_t subset_idx = ch / channel_stride_;
                x_1_subset_[m_id][subset_idx] = x_1_[m_id][ch];
                y_1_subset_[m_id][subset_idx] = y_1_[m_id][ch];
                z_1_subset_[m_id][subset_idx] = z_1_[m_id][ch];
            }
        }
    }

    // Pre-allocate memory for both buffers for performance
    data_buffer1_ = std::make_unique<LidarFrame>();
    data_buffer2_ = std::make_unique<LidarFrame>();
    data_buffer1_->reserve(columns_per_frame_ * pixels_per_column_);
    data_buffer2_->reserve(columns_per_frame_ * pixels_per_column_);

    Eigen::Vector3f half_dims = vehicle_box_dimensions_ / 2.0f;
    vehicle_box_min_ = vehicle_box_center_ - half_dims;
    vehicle_box_max_ = vehicle_box_center_ + half_dims;

    // Sanity checks for lookup table sizes
    if (x_1_.size() != static_cast<size_t>(columns_per_frame_) || (!x_1_.empty() && x_1_[0].size() != static_cast<size_t>(pixels_per_column_))) {
        throw std::runtime_error("x_1_ lookup table size mismatch after initialization");
    }
    if (x_1_subset_.size() != static_cast<size_t>(columns_per_frame_) || (!x_1_subset_.empty() && x_1_subset_[0].size() != static_cast<size_t>(subset_channels_))) {
        throw std::runtime_error("x_1_subset_ lookup table size mismatch after initialization");
    }
    if (x_2_.size() != static_cast<size_t>(columns_per_frame_)) {
        throw std::runtime_error("x_2_ lookup table size mismatch after initialization");
    }
    if (pixel_shifts_.size() != static_cast<size_t>(pixels_per_column_)) {
        throw std::runtime_error("pixel_shifts_ size mismatch after initialization");
    }
    if (pixel_shifts_subset_.size() != static_cast<size_t>(subset_channels_)) {
        throw std::runtime_error("pixel_shifts_subset_ size mismatch after initialization");
    }

#ifdef DEBUG
    if (!x_1_.empty() && !x_1_[0].empty()) {
        assert(reinterpret_cast<uintptr_t>(x_1_[0].data()) % 32 == 0 && "x_1_[0].data() not 32-byte aligned!");
    }
    if (!x_1_subset_.empty() && !x_1_subset_[0].empty()) {
        assert(reinterpret_cast<uintptr_t>(x_1_subset_[0].data()) % 32 == 0 && "x_1_subset_[0].data() not 32-byte aligned!");
    }
#endif
}
// %            ... decode_packet_legacy
// %            ... decode_packet_legacy
std::unique_ptr<LidarFrame> LidarCallback::DecodePacketLegacy(const std::vector<uint8_t>& packet) {
    if (packet.size() != expected_size_) {
        std::cerr << "Invalid packet size: " << packet.size() << ", expected: " << expected_size_ << std::endl;
        return nullptr;
    }

    LidarFrame* p_current_write_buffer = buffer_toggle_ ? data_buffer2_.get() : data_buffer1_.get();

    double prev_frame_completed_latest_ts = 0.0;
    if (this->latest_timestamp_s > 0.0) {
        prev_frame_completed_latest_ts = this->latest_timestamp_s;
    }

    for (int col = 0; col < columns_per_packet_; ++col) {
        size_t block_offset = col * block_size_;

        uint64_t timestamp_ns_raw;
        std::memcpy(&timestamp_ns_raw, packet.data() + block_offset, sizeof(uint64_t));
        uint64_t timestamp_ns = le64toh(timestamp_ns_raw);
        double current_col_timestamp_s = std::fmod(static_cast<double>(timestamp_ns) * 1e-9, 86400.0);

        if (current_col_timestamp_s < 0) {
            std::cerr << "Negative column timestamp: " << current_col_timestamp_s << std::endl;
            continue;
        }

        uint16_t m_id_raw;
        std::memcpy(&m_id_raw, packet.data() + block_offset + 8, sizeof(uint16_t));
        uint16_t m_id = le16toh(m_id_raw);

        if (m_id >= static_cast<uint16_t>(columns_per_frame_)) {
            std::cerr << "Invalid measurement ID: " << m_id << " (>= " << columns_per_frame_ << ")" << std::endl;
            continue;
        }

        uint16_t current_packet_frame_id_raw;
        std::memcpy(&current_packet_frame_id_raw, packet.data() + block_offset + 10, sizeof(uint16_t));
        uint16_t current_packet_frame_id = le16toh(current_packet_frame_id_raw);

        if (current_packet_frame_id != this->frame_id_) {
            if (this->frame_id_ != 0 || this->number_points_ > 0) {
                p_current_write_buffer->numberpoints = this->number_points_;
                p_current_write_buffer->timestamp_end = this->latest_timestamp_s;
            }
            prev_frame_completed_latest_ts = this->latest_timestamp_s;
            SwapBuffer();
            p_current_write_buffer = buffer_toggle_ ? data_buffer2_.get() : data_buffer1_.get();
            this->number_points_ = 0;
            this->frame_id_ = current_packet_frame_id;
            p_current_write_buffer->clear();
            p_current_write_buffer->reserve(columns_per_frame_ * pixels_per_column_);
        }

        this->latest_timestamp_s = current_col_timestamp_s;

        uint32_t block_status;
        size_t status_offset = block_offset + 16 + (pixels_per_column_ * 12);
        std::memcpy(&block_status, packet.data() + status_offset, sizeof(uint32_t));
        block_status = le32toh(block_status);
        if (block_status != 0xFFFFFFFF) {
            continue;
        }

        bool is_first_point_of_current_frame = (this->number_points_ == 0);

#ifdef __AVX2__
        for (uint16_t subset_idx_base = 0; subset_idx_base < static_cast<uint16_t>(subset_channels_); subset_idx_base += 8) {
            size_t first_pixel_in_block_offset = block_offset + 16;

            alignas(32) float range_m[8];
            alignas(32) float r_min_vals[8];
            alignas(32) float r_max_vals[8];
            uint16_t c_ids[8];
            uint8_t reflectivity[8];
            uint16_t signal[8], nir[8];

            for (int i = 0; i < 8; ++i) {
                uint16_t subset_idx = subset_idx_base + i;
                if (subset_idx >= static_cast<uint16_t>(subset_channels_)) {
                    range_m[i] = 0.0f;
                    r_min_vals[i] = 1.0f;
                    r_max_vals[i] = 0.0f;
                    c_ids[i] = 0;
                    continue;
                }

                uint16_t current_c_id = subset_c_ids_[subset_idx];
                size_t pixel_data_offset = first_pixel_in_block_offset + current_c_id * 12;
                if (pixel_data_offset + 11 >= packet.size()) {
                    range_m[i] = 0.0f;
                    r_min_vals[i] = 1.0f;
                    r_max_vals[i] = 0.0f;
                    c_ids[i] = 0;
                    continue;
                }

                uint32_t range_mm_raw;
                std::memcpy(&range_mm_raw, packet.data() + pixel_data_offset, sizeof(uint32_t));
                uint32_t range_mm = le32toh(range_mm_raw) & 0x000FFFFF;
                range_m[i] = static_cast<float>(range_mm) * 0.001f;
                r_min_vals[i] = r_min_subset_[subset_idx];
                r_max_vals[i] = r_max_subset_[subset_idx];
                c_ids[i] = current_c_id;

                std::memcpy(&reflectivity[i], packet.data() + pixel_data_offset + 4, sizeof(uint8_t));
                uint16_t signal_raw, nir_raw;
                std::memcpy(&signal_raw, packet.data() + pixel_data_offset + 6, sizeof(uint16_t));
                std::memcpy(&nir_raw, packet.data() + pixel_data_offset + 8, sizeof(uint16_t));
                signal[i] = le16toh(signal_raw);
                nir[i] = le16toh(nir_raw);
            }

            __m256 m256_range = _mm256_load_ps(range_m);
            __m256 m256_r_min_vec = _mm256_load_ps(r_min_vals);
            __m256 m256_r_max_vec = _mm256_load_ps(r_max_vals);
            __m256 min_mask = _mm256_cmp_ps(m256_range, m256_r_min_vec, _CMP_GE_OQ);
            __m256 max_mask = _mm256_cmp_ps(m256_range, m256_r_max_vec, _CMP_LE_OQ);
            __m256 valid_mask = _mm256_and_ps(min_mask, max_mask);

            __m256 x1_vec = _mm256_load_ps(x_1_subset_[m_id].data() + subset_idx_base);
            __m256 y1_vec = _mm256_load_ps(y_1_subset_[m_id].data() + subset_idx_base);
            __m256 z1_vec = _mm256_load_ps(z_1_subset_[m_id].data() + subset_idx_base);
            __m256 x2_val = _mm256_set1_ps(x_2_[m_id]);
            __m256 y2_val = _mm256_set1_ps(y_2_[m_id]);
            __m256 z2_val = _mm256_set1_ps(z_2_[m_id]);

            __m256 pt_x = _mm256_fmadd_ps(m256_range, x1_vec, x2_val);
            __m256 pt_y = _mm256_fmadd_ps(m256_range, y1_vec, y2_val);
            __m256 pt_z = _mm256_fmadd_ps(m256_range, z1_vec, z2_val);

            alignas(32) float pt_x_arr[8], pt_y_arr[8], pt_z_arr[8];
            _mm256_store_ps(pt_x_arr, pt_x);
            _mm256_store_ps(pt_y_arr, pt_y);
            _mm256_store_ps(pt_z_arr, pt_z);

            alignas(32) float valid_mask_arr[8];
            _mm256_store_ps(valid_mask_arr, valid_mask);

            double relative_timestamp_s = (p_current_write_buffer->numberpoints > 0 || this->number_points_ > 0) && p_current_write_buffer->timestamp > 0
                ? std::max(0.0, current_col_timestamp_s - p_current_write_buffer->timestamp)
                : 0.0;

            for (int i = 0; i < 8; ++i) {
                uint16_t subset_idx = subset_idx_base + i;
                if (subset_idx >= static_cast<uint16_t>(subset_channels_)) break;
                if (range_m[i] >= r_min_vals[i] && range_m[i] <= r_max_vals[i] && range_m[i] > 0) {

                    bool is_in_vehicle_box = (pt_x_arr[i] >= vehicle_box_min_.x() && pt_x_arr[i] <= vehicle_box_max_.x() &&
                                  pt_y_arr[i] >= vehicle_box_min_.y() && pt_y_arr[i] <= vehicle_box_max_.y() &&
                                  pt_z_arr[i] >= vehicle_box_min_.z() && pt_z_arr[i] <= vehicle_box_max_.z());

                    // Check if the Z coordinate is within the desired range [-200.0, 0.0]
                    if (!is_in_vehicle_box && ((pt_z_arr[i] >= zfiltermin_ && pt_z_arr[i] <= zfiltermax_) || (reflectivity[i] >= reflectivity_threshold_))) {
                        p_current_write_buffer->x.push_back(pt_x_arr[i]);
                        p_current_write_buffer->y.push_back(pt_y_arr[i]);
                        p_current_write_buffer->z.push_back(pt_z_arr[i]);
                        p_current_write_buffer->c_id.push_back(c_ids[i]);
                        p_current_write_buffer->m_id.push_back(m_id);
                        p_current_write_buffer->timestamp_points.push_back(current_col_timestamp_s);
                        p_current_write_buffer->relative_timestamp.push_back(static_cast<float>(relative_timestamp_s));
                        p_current_write_buffer->reflectivity.push_back(reflectivity[i]);
                        p_current_write_buffer->signal.push_back(signal[i]);
                        p_current_write_buffer->nir.push_back(nir[i]);

                        this->number_points_++;
                        if (is_first_point_of_current_frame) {
                            p_current_write_buffer->timestamp = current_col_timestamp_s;
                            p_current_write_buffer->frame_id = this->frame_id_;
                            p_current_write_buffer->interframe_timedelta = (prev_frame_completed_latest_ts > 0.0)
                                ? std::max(0.0, current_col_timestamp_s - prev_frame_completed_latest_ts) : 0.0;
                            is_first_point_of_current_frame = false;
                        }
                    }
                }
            }
        }
#else
        size_t first_pixel_in_block_offset = block_offset + 16;
        for (uint16_t c_id = 0; c_id < static_cast<uint16_t>(pixels_per_column_); c_id += channel_stride_) {
            size_t pixel_data_offset = first_pixel_in_block_offset + c_id * 12;
            if (pixel_data_offset + 11 >= packet.size()) {
                continue;
            }

            uint32_t range_mm_raw;
            std::memcpy(&range_mm_raw, packet.data() + pixel_data_offset, sizeof(uint32_t));
            uint32_t range_mm = le32toh(range_mm_raw) & 0x000FFFFF;
            float range_m = static_cast<float>(range_mm) * 0.001f;

            if (range_m < r_min_[c_id] || range_m > r_max_[c_id] || range_m == 0) {
                continue;
            }

            uint8_t current_reflectivity;
            std::memcpy(&current_reflectivity, packet.data() + pixel_data_offset + 4, sizeof(uint8_t));
            uint16_t signal_raw, nir_raw;
            std::memcpy(&signal_raw, packet.data() + pixel_data_offset + 6, sizeof(uint16_t));
            std::memcpy(&nir_raw, packet.data() + pixel_data_offset + 8, sizeof(uint16_t));
            uint16_t current_signal = le16toh(signal_raw);
            uint16_t current_nir = le16toh(nir_raw);

            float pt_x = range_m * x_1_[m_id][c_id] + x_2_[m_id];
            float pt_y = range_m * y_1_[m_id][c_id] + y_2_[m_id];
            float pt_z = range_m * z_1_[m_id][c_id] + z_2_[m_id];

            bool is_in_vehicle_box = (pt_x >= vehicle_box_min_.x() && pt_x <= vehicle_box_max_.x() &&
                                     pt_y >= vehicle_box_min_.y() && pt_y <= vehicle_box_max_.y() &&
                                     pt_z >= vehicle_box_min_.z() && pt_z <= vehicle_box_max_.z());

            // Check if the Z coordinate is within the desired range [-200.0, 0.0]
            if (!is_in_vehicle_box && ((pt_z >= zfiltermin_ && pt_z <= zfiltermax_) || (current_reflectivity >= reflectivity_threshold_))) { // <-- ADDED FILTER
                double relative_timestamp_s = (p_current_write_buffer->numberpoints > 0 || this->number_points_ > 0) && p_current_write_buffer->timestamp > 0
                    ? std::max(0.0, current_col_timestamp_s - p_current_write_buffer->timestamp)
                    : 0.0;

                p_current_write_buffer->x.push_back(pt_x);
                p_current_write_buffer->y.push_back(pt_y);
                p_current_write_buffer->z.push_back(pt_z);
                p_current_write_buffer->c_id.push_back(c_id);
                p_current_write_buffer->m_id.push_back(m_id);
                p_current_write_buffer->timestamp_points.push_back(current_col_timestamp_s);
                p_current_write_buffer->relative_timestamp.push_back(static_cast<float>(relative_timestamp_s));
                p_current_write_buffer->reflectivity.push_back(current_reflectivity);
                p_current_write_buffer->signal.push_back(current_signal);
                p_current_write_buffer->nir.push_back(current_nir);

                this->number_points_++;
                if (is_first_point_of_current_frame) {
                    p_current_write_buffer->timestamp = current_col_timestamp_s;
                    p_current_write_buffer->frame_id = this->frame_id_;
                    p_current_write_buffer->interframe_timedelta = (prev_frame_completed_latest_ts > 0.0)
                        ? std::max(0.0, current_col_timestamp_s - prev_frame_completed_latest_ts) : 0.0;
                    is_first_point_of_current_frame = false;
                }
            }
        }
#endif
    }

    if (p_current_write_buffer) {
        p_current_write_buffer->numberpoints = this->number_points_;
    }
    return std::make_unique<LidarFrame>(GetLatestFrame());
}
// %            ... decode_packet_single_return
std::unique_ptr<LidarFrame> LidarCallback::DecodePacketRng19(const std::vector<uint8_t>& packet) {
    if (packet.size() != expected_size_) {
        std::cerr << "Invalid packet size: " << packet.size() << ", expected: " << expected_size_ << std::endl;
        return nullptr;
    }

    uint16_t packet_type_raw;
    std::memcpy(&packet_type_raw, packet.data(), sizeof(uint16_t));
    uint16_t packet_type = le16toh(packet_type_raw);
    if (packet_type != 0x0001) {
        std::cerr << "Invalid packet type: 0x" << std::hex << packet_type << std::dec << " (expected 0x1)" << std::endl;
        return nullptr;
    }

    uint16_t current_packet_frame_id_raw;
    std::memcpy(&current_packet_frame_id_raw, packet.data() + 2, sizeof(uint16_t));
    uint16_t current_packet_frame_id = le16toh(current_packet_frame_id_raw);

    LidarFrame* p_current_write_buffer = buffer_toggle_ ? data_buffer2_.get() : data_buffer1_.get();

    double prev_frame_completed_latest_ts = 0.0;
    if (current_packet_frame_id != this->frame_id_) {
        if (this->frame_id_ != 0 || this->number_points_ > 0) {
            p_current_write_buffer->numberpoints = this->number_points_;
            p_current_write_buffer->timestamp_end = this->latest_timestamp_s;
            std::unique_ptr<LidarFrame> completed_frame = std::move(buffer_toggle_ ? data_buffer2_ : data_buffer1_);
        }
        prev_frame_completed_latest_ts = this->latest_timestamp_s;
        SwapBuffer();
        p_current_write_buffer = buffer_toggle_ ? data_buffer2_.get() : data_buffer1_.get();
        this->number_points_ = 0;
        this->frame_id_ = current_packet_frame_id;
        p_current_write_buffer->clear();
        p_current_write_buffer->reserve(columns_per_frame_ * pixels_per_column_);
    }

    bool is_first_point_of_current_frame = (this->number_points_ == 0);

    for (int col = 0; col < columns_per_packet_; ++col) {
        size_t block_offset = PACKET_HEADER_BYTES + col * block_size_;

        uint64_t timestamp_ns_raw;
        std::memcpy(&timestamp_ns_raw, packet.data() + block_offset, sizeof(uint64_t));
        uint64_t timestamp_ns = le64toh(timestamp_ns_raw);
        double current_col_timestamp_s = std::fmod(static_cast<double>(timestamp_ns) * 1e-9, 86400.0);

        if (current_col_timestamp_s < 0) {
            std::cerr << "Negative column timestamp: " << current_col_timestamp_s << std::endl;
            continue;
        }
        this->latest_timestamp_s = current_col_timestamp_s;

        uint16_t m_id_raw;
        std::memcpy(&m_id_raw, packet.data() + block_offset + 8, sizeof(uint16_t));
        uint16_t m_id = le16toh(m_id_raw);

        if (m_id >= static_cast<uint16_t>(columns_per_frame_)) {
            std::cerr << "Invalid measurement ID: " << m_id << " (>= " << columns_per_frame_ << ")" << std::endl;
            continue;
        }

        uint8_t column_status;
        std::memcpy(&column_status, packet.data() + block_offset + 10, sizeof(uint8_t));
        if (!(column_status & 0x01)) {
            continue;
        }

#ifdef __AVX2__
        for (uint16_t subset_idx_base = 0; subset_idx_base < static_cast<uint16_t>(subset_channels_); subset_idx_base += 8) {
            size_t first_pixel_in_block_offset = block_offset + 12;

            alignas(32) float range_m[8];
            alignas(32) float r_min_vals[8];
            alignas(32) float r_max_vals[8];
            uint16_t c_ids[8];
            uint8_t reflectivity[8];
            uint16_t signal[8], nir[8];

            for (int i = 0; i < 8; ++i) {
                uint16_t subset_idx = subset_idx_base + i;
                if (subset_idx >= static_cast<uint16_t>(subset_channels_)) {
                    range_m[i] = 0.0f;
                    r_min_vals[i] = 1.0f;
                    r_max_vals[i] = 0.0f;
                    c_ids[i] = 0;
                    continue;
                }

                uint16_t current_c_id = subset_c_ids_[subset_idx];
                size_t pixel_data_offset = first_pixel_in_block_offset + current_c_id * 12;
                if (pixel_data_offset + 11 >= packet.size()) {
                    range_m[i] = 0.0f;
                    r_min_vals[i] = 1.0f;
                    r_max_vals[i] = 0.0f;
                    c_ids[i] = 0;
                    continue;
                }

                uint32_t range_mm_raw;
                uint8_t range_bytes[4] = {packet[pixel_data_offset], packet[pixel_data_offset + 1], packet[pixel_data_offset + 2], 0};
                std::memcpy(&range_mm_raw, range_bytes, sizeof(uint32_t));
                uint32_t range_mm = le32toh(range_mm_raw) & 0x0007FFFF;
                range_m[i] = static_cast<float>(range_mm) * 0.001f;
                r_min_vals[i] = r_min_subset_[subset_idx];
                r_max_vals[i] = r_max_subset_[subset_idx];
                c_ids[i] = current_c_id;

                std::memcpy(&reflectivity[i], packet.data() + pixel_data_offset + 4, sizeof(uint8_t));
                uint16_t signal_raw, nir_raw;
                std::memcpy(&signal_raw, packet.data() + pixel_data_offset + 6, sizeof(uint16_t));
                std::memcpy(&nir_raw, packet.data() + pixel_data_offset + 8, sizeof(uint16_t));
                signal[i] = le16toh(signal_raw);
                nir[i] = le16toh(nir_raw);
            }

            __m256 m256_range = _mm256_load_ps(range_m);
            __m256 m256_r_min_vec = _mm256_load_ps(r_min_vals);
            __m256 m256_r_max_vec = _mm256_load_ps(r_max_vals);
            __m256 min_mask = _mm256_cmp_ps(m256_range, m256_r_min_vec, _CMP_GE_OQ);
            __m256 max_mask = _mm256_cmp_ps(m256_range, m256_r_max_vec, _CMP_LE_OQ);
            __m256 valid_mask = _mm256_and_ps(min_mask, max_mask);

            __m256 x1_vec = _mm256_load_ps(x_1_subset_[m_id].data() + subset_idx_base);
            __m256 y1_vec = _mm256_load_ps(y_1_subset_[m_id].data() + subset_idx_base);
            __m256 z1_vec = _mm256_load_ps(z_1_subset_[m_id].data() + subset_idx_base);
            __m256 x2_val = _mm256_set1_ps(x_2_[m_id]);
            __m256 y2_val = _mm256_set1_ps(y_2_[m_id]);
            __m256 z2_val = _mm256_set1_ps(z_2_[m_id]);

            __m256 pt_x = _mm256_fmadd_ps(m256_range, x1_vec, x2_val);
            __m256 pt_y = _mm256_fmadd_ps(m256_range, y1_vec, y2_val);
            __m256 pt_z = _mm256_fmadd_ps(m256_range, z1_vec, z2_val);

            alignas(32) float pt_x_arr[8], pt_y_arr[8], pt_z_arr[8];
            _mm256_store_ps(pt_x_arr, pt_x);
            _mm256_store_ps(pt_y_arr, pt_y);
            _mm256_store_ps(pt_z_arr, pt_z);

            alignas(32) float valid_mask_arr[8];
            _mm256_store_ps(valid_mask_arr, valid_mask);

            double relative_timestamp_s = (p_current_write_buffer->numberpoints > 0 || this->number_points_ > 0) && p_current_write_buffer->timestamp > 0
                ? std::max(0.0, current_col_timestamp_s - p_current_write_buffer->timestamp)
                : 0.0;

            for (int i = 0; i < 8; ++i) {
                uint16_t subset_idx = subset_idx_base + i;
                if (subset_idx >= static_cast<uint16_t>(subset_channels_)) break;
                if (range_m[i] >= r_min_vals[i] && range_m[i] <= r_max_vals[i] && range_m[i] > 0) {
                    // Check if the Z coordinate is within the desired range [-200.0, 0.0]
                    bool is_in_vehicle_box = (pt_x_arr[i] >= vehicle_box_min_.x() && pt_x_arr[i] <= vehicle_box_max_.x() &&
                                  pt_y_arr[i] >= vehicle_box_min_.y() && pt_y_arr[i] <= vehicle_box_max_.y() &&
                                  pt_z_arr[i] >= vehicle_box_min_.z() && pt_z_arr[i] <= vehicle_box_max_.z());

                    if (!is_in_vehicle_box && ((pt_z_arr[i] >= zfiltermin_ && pt_z_arr[i] <= zfiltermax_) || (reflectivity[i] >= reflectivity_threshold_))) {
                        p_current_write_buffer->x.push_back(pt_x_arr[i]);
                        p_current_write_buffer->y.push_back(pt_y_arr[i]);
                        p_current_write_buffer->z.push_back(pt_z_arr[i]);
                        p_current_write_buffer->c_id.push_back(c_ids[i]);
                        p_current_write_buffer->m_id.push_back(m_id);
                        p_current_write_buffer->timestamp_points.push_back(current_col_timestamp_s);
                        p_current_write_buffer->relative_timestamp.push_back(static_cast<float>(relative_timestamp_s));
                        p_current_write_buffer->reflectivity.push_back(reflectivity[i]);
                        p_current_write_buffer->signal.push_back(signal[i]);
                        p_current_write_buffer->nir.push_back(nir[i]);

                        this->number_points_++;
                        if (is_first_point_of_current_frame) {
                            p_current_write_buffer->timestamp = current_col_timestamp_s;
                            p_current_write_buffer->frame_id = this->frame_id_;
                            p_current_write_buffer->interframe_timedelta = (prev_frame_completed_latest_ts > 0.0)
                                ? std::max(0.0, current_col_timestamp_s - prev_frame_completed_latest_ts) : 0.0;
                            is_first_point_of_current_frame = false;
                        }
                    }
                }
            }
        }
#else
        size_t first_pixel_in_block_offset = block_offset + 12;
        for (uint16_t c_id = 0; c_id < static_cast<uint16_t>(pixels_per_column_); c_id += channel_stride_) {
            size_t pixel_data_offset = first_pixel_in_block_offset + c_id * 12;
            if (pixel_data_offset + 11 >= packet.size()) {
                continue;
            }

            uint32_t range_mm_raw;
            uint8_t range_bytes[4] = {packet[pixel_data_offset], packet[pixel_data_offset + 1], packet[pixel_data_offset + 2], 0};
            std::memcpy(&range_mm_raw, range_bytes, sizeof(uint32_t));
            uint32_t range_mm = le32toh(range_mm_raw) & 0x0007FFFF;
            float range_m = static_cast<float>(range_mm) * 0.001f;

            if (range_m < r_min_[c_id] || range_m > r_max_[c_id] || range_m == 0) {
                continue;
            }

            uint8_t current_reflectivity;
            std::memcpy(&current_reflectivity, packet.data() + pixel_data_offset + 4, sizeof(uint8_t));
            uint16_t signal_raw, nir_raw;
            std::memcpy(&signal_raw, packet.data() + pixel_data_offset + 6, sizeof(uint16_t));
            std::memcpy(&nir_raw, packet.data() + pixel_data_offset + 8, sizeof(uint16_t));
            uint16_t current_signal = le16toh(signal_raw);
            uint16_t current_nir = le16toh(nir_raw);

            float pt_x = range_m * x_1_[m_id][c_id] + x_2_[m_id];
            float pt_y = range_m * y_1_[m_id][c_id] + y_2_[m_id];
            float pt_z = range_m * z_1_[m_id][c_id] + z_2_[m_id];

            bool is_in_vehicle_box = (pt_x >= vehicle_box_min_.x() && pt_x <= vehicle_box_max_.x() &&
                                     pt_y >= vehicle_box_min_.y() && pt_y <= vehicle_box_max_.y() &&
                                     pt_z >= vehicle_box_min_.z() && pt_z <= vehicle_box_max_.z());

            // Check if the Z coordinate is within the desired range [-200.0, 0.0]
            if (!is_in_vehicle_box && ((pt_z >= zfiltermin_ && pt_z <= zfiltermax_) || (current_reflectivity >= reflectivity_threshold_))) {
                double relative_timestamp_s = (p_current_write_buffer->numberpoints > 0 || this->number_points_ > 0) && p_current_write_buffer->timestamp > 0
                    ? std::max(0.0, current_col_timestamp_s - p_current_write_buffer->timestamp)
                    : 0.0;

                p_current_write_buffer->x.push_back(pt_x);
                p_current_write_buffer->y.push_back(pt_y);
                p_current_write_buffer->z.push_back(pt_z);
                p_current_write_buffer->c_id.push_back(c_id);
                p_current_write_buffer->m_id.push_back(m_id);
                p_current_write_buffer->timestamp_points.push_back(current_col_timestamp_s);
                p_current_write_buffer->relative_timestamp.push_back(static_cast<float>(relative_timestamp_s));
                p_current_write_buffer->reflectivity.push_back(current_reflectivity);
                p_current_write_buffer->signal.push_back(current_signal);
                p_current_write_buffer->nir.push_back(current_nir);

                this->number_points_++;
                if (is_first_point_of_current_frame) {
                    p_current_write_buffer->timestamp = current_col_timestamp_s;
                    p_current_write_buffer->frame_id = this->frame_id_;
                    p_current_write_buffer->interframe_timedelta = (prev_frame_completed_latest_ts > 0.0)
                        ? std::max(0.0, current_col_timestamp_s - prev_frame_completed_latest_ts) : 0.0;
                    is_first_point_of_current_frame = false;
                }
            }
        }
#endif
    }

    if (p_current_write_buffer) {
        p_current_write_buffer->numberpoints = this->number_points_;
    }
    return std::make_unique<LidarFrame>(GetLatestFrame());
}