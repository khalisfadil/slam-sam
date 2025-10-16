// OusterLidarCallback.hpp
#pragma once

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <cstdint>
#include <mutex>

#include <dataframe.hpp>

class LidarCallback {
    public:

        explicit LidarCallback(const std::string& json_meta_path, const std::string& json_param_path);
        explicit LidarCallback(const nlohmann::json& json_meta,const nlohmann::json& json_param);
        void DecodePacketRng19(const std::vector<uint8_t>& packet, LidarFrame& frame);
        void DecodePacketLegacy(const std::vector<uint8_t>& packet, LidarFrame& frame); 
        const LidarFrame& GetLatestFrame() const { return buffer_toggle_ ? data_buffer1_ : data_buffer2_; }
        const nlohmann::json& GetMetadata() const { return metadata_; }

    private:

        nlohmann::json metadata_;
        nlohmann::json parameter_;

        Eigen::Matrix3d body_to_lidar_rotation_ = Eigen::Matrix3d::Zero();
        Eigen::Vector3d body_to_lidar_translation_ = Eigen::Vector3d::Zero();
        uint16_t channel_stride_ = 1; // Number of rows to skip (N), default to 1 (process all rows)
        uint16_t subset_channels_;    // Number of channels in subset tables (ceiling(pixels_per_column_ / N))

        // Original lookup tables
        std::vector<std::vector<float, Eigen::aligned_allocator<float>>> x_1_;
        std::vector<std::vector<float, Eigen::aligned_allocator<float>>> y_1_;
        std::vector<std::vector<float, Eigen::aligned_allocator<float>>> z_1_;
        std::vector<float> x_2_;
        std::vector<float> y_2_;
        std::vector<float> z_2_;
        std::vector<float> r_min_;
        std::vector<float> r_max_;
        std::vector<float> sin_beam_azimuths_;
        std::vector<float> cos_beam_azimuths_;
        std::vector<float> sin_beam_altitudes_;
        std::vector<float> cos_beam_altitudes_;
        std::vector<int> pixel_shifts_;
        // Subset lookup tables for channels that are multiples of N
        std::vector<std::vector<float, Eigen::aligned_allocator<float>>> x_1_subset_;
        std::vector<std::vector<float, Eigen::aligned_allocator<float>>> y_1_subset_;
        std::vector<std::vector<float, Eigen::aligned_allocator<float>>> z_1_subset_;
        std::vector<float> r_min_subset_;
        std::vector<float> r_max_subset_;
        std::vector<float> sin_beam_azimuths_subset_;
        std::vector<float> cos_beam_azimuths_subset_;
        std::vector<float> sin_beam_altitudes_subset_;
        std::vector<float> cos_beam_altitudes_subset_;
        std::vector<int> pixel_shifts_subset_;
        std::vector<uint16_t> subset_c_ids_; // Maps subset indices to original c_id

        Eigen::Matrix4d lidar_to_sensor_transform_;
        float lidar_origin_to_beam_origin_mm_;
        size_t block_size_;
        size_t expected_size_;
        size_t PACKET_HEADER_BYTES = 32;
        size_t PACKET_FOOTER_BYTES = 32;
        size_t COLUMN_HEADER_BYTES = 12;
        size_t CHANNEL_STRIDE_BYTES = 12;
        size_t MEASUREMENT_BLOCK_STATUS_BYTES = 0;
        std::string udp_profile_lidar_ = "UNKNOWN";
        int columns_per_frame_ = 2048;
        int pixels_per_column_ = 128;
        int columns_per_packet_ = 16;
        uint16_t frame_id_ = 0;
        uint32_t number_points_ = 0;
        double latest_timestamp_s = 0.0;
        LidarFrame data_buffer1_;
        LidarFrame data_buffer2_;
        bool buffer_toggle_ = true;
        float zfiltermax_ = 0.0f;
        float zfiltermin_ = -300.0f;
        float rfiltermax_ = 200.0f;
        float rfiltermin_ = 1.0f;
        uint8_t reflectivity_threshold_ = 0;

        void Initialize();
        void ParseMetadata(const nlohmann::json& json_data);
        void ParseParamdata(const nlohmann::json& json_data);
        void SwapBuffer() { buffer_toggle_ = !buffer_toggle_; }
};