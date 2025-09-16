#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include<dataframe.hpp>

class CompCallback {
    public:
        explicit CompCallback(const std::string& json_path);
        explicit CompCallback(const nlohmann::json& json_data);
        void Decode(const std::vector<uint8_t>& packet, CompFrame& frame);
    private:

        nlohmann::json metadata_;
        const double E = 5.2185400842339e5;
        const double E2 = E * E;
        const double GM = 3986004.418e8;
        const double a = 6378137.0;
        const double b = 6356752.3142;
        const double e2 = 6.69437999014e-3;
        const double b_over_a = 0.996647189335;
        const double omega = 7.292115e-5;                                   
        double updateRate_ = 100;                                               // frequency rate Hz
        Eigen::Vector3d velocityRandomWalk_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d angularRandomWalk_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d biasAccelerometer_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d biasGyroscope_ = Eigen::Vector3d::Zero();
        Eigen::Matrix3d body_to_imu_rotation_ = Eigen::Matrix3d::Zero();
        Eigen::Vector3d body_to_imu_translation_ = Eigen::Vector3d::Zero();

        void ParseMetadata(const nlohmann::json& json_data);
        double GravityWGS84(double latitude, double longitude, double altitude);
};
