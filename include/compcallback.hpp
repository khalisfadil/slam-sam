#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include <dataframe.hpp>
#include <gtsam/geometry/Point3.h>

#include <mutex>
#include <deque>

class CompCallback {
    public:
        explicit CompCallback(const std::string& json_path);
        explicit CompCallback(const nlohmann::json& json_data);
        std::unique_ptr<CompFrame> DecodePacket(const std::vector<uint8_t>& packet);
        void ReturnFrameToPool(std::unique_ptr<CompFrame> frame);

        double GravityWGS84(double latitude, double longitude, double altitude);
        gtsam::Vector3 getStaticBiasAccelerometer() const { return Eigen::Vector3d(staticBiasAccelerometer_).cast<gtsam::Vector::Scalar>(); }
        gtsam::Vector3 getStaticBiasGyroscope() const { return Eigen::Vector3d(staticBiasGyroscope_).cast<gtsam::Vector::Scalar>(); }
        gtsam::Vector3 getVelocityRandomWalk() const { return Eigen::Vector3d(velocityRandomWalk_).cast<gtsam::Vector::Scalar>(); }
        gtsam::Vector3 getAngularRandomWalk() const { return Eigen::Vector3d(angularRandomWalk_).cast<gtsam::Vector::Scalar>(); }
        gtsam::Vector3 getBiasInstabilityAccelerometer() const { return Eigen::Vector3d(biasInstabilityAccelerometer_).cast<gtsam::Vector::Scalar>(); }
        gtsam::Vector3 getBiasInstabilityGyroscope() const { return Eigen::Vector3d(biasInstabilityGyroscope_).cast<gtsam::Vector::Scalar>(); }
        gtsam::Vector3 getBiasRandomWalkAccelerometer() const { return Eigen::Vector3d(biasRandomWalkAccelerometer_).cast<gtsam::Vector::Scalar>(); } 
        gtsam::Vector3 getBiasRandomWalkGyroscope() const { return Eigen::Vector3d(biasRandomWalkGyroscope_).cast<gtsam::Vector::Scalar>(); }     
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
        double updateRate_ = 50;        
        size_t poolsize_ = 4;
        Eigen::Vector3d staticBiasAccelerometer_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d staticBiasGyroscope_ = Eigen::Vector3d::Zero();                                       // frequency rate Hz
        Eigen::Vector3d velocityRandomWalk_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d angularRandomWalk_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d biasInstabilityAccelerometer_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d biasInstabilityGyroscope_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d biasRandomWalkAccelerometer_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d biasRandomWalkGyroscope_ = Eigen::Vector3d::Zero();
        Eigen::Matrix3d body_to_imu_rotation_ = Eigen::Matrix3d::Zero();
        Eigen::Vector3d body_to_imu_translation_ = Eigen::Vector3d::Zero();
        std::unique_ptr<CompFrame> active_frame_;
        std::deque<std::unique_ptr<CompFrame>> frame_pool_;
        std::mutex pool_mutex_;

        void InitializePool(size_t pool_size = 4);
        std::unique_ptr<CompFrame> GetFrameFromPool();
        void ParseMetadata(const nlohmann::json& json_data);
};
