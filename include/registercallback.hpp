#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <dataframe.hpp>

#include <pclomp/ndt_omp.h>
#include <pclomp/ndt_omp_impl.hpp>
#include <pclomp/voxel_grid_covariance_omp.h>
#include <pclomp/voxel_grid_covariance_omp_impl.hpp>
#include <pclomp/gicp_omp.h>
#include <pclomp/gicp_omp_impl.hpp>

#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>

class RegisterCallback {
    public:
        explicit RegisterCallback(const std::string& json_param_path);
        explicit RegisterCallback(const nlohmann::json& json_param);
        Eigen::Vector3d lla2ned(double lat, double lon, double alt, double rlat, double rlon, double ralt);
        Eigen::Vector3d ned2lla(double n, double e, double d, double rlat, double rlon, double ralt);
        double SymmetricalAngle(double x);
        Eigen::Matrix3f Cb2n(const Eigen::Quaternionf& q) ;
        Eigen::Matrix<double, 6, 6> reorderCovarianceForGTSAM(const Eigen::Matrix<double, 6, 6>& ndt_covariance);


        pcl::Registration <pcl::PointXYZI, pcl::PointXYZI>::Ptr registration;

        int num_threads_ = 8;
        float mapvoxelsize_ = 1.0;
        std::string registration_method_ = "NDT_OMP";
        float ndt_resolution_ = 1.0;
        float ndt_transform_epsilon_ = 0.01;
        std::string ndt_neighborhood_search_method_ = "DIRECT7";
        float gicp_corr_dist_threshold_ = 5.0;
        float gicp_transform_epsilon_ = 0.01;
    private:
        nlohmann::json parameter_;

        void ParseParamdata(const nlohmann::json& json_data);
};