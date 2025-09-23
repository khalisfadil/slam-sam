#include <registercallback.hpp>

using json = nlohmann::json;
// %            ... initgravity
RegisterCallback::RegisterCallback(const std::string& json_param_path) {
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

}
// %            ... initgravity
RegisterCallback::RegisterCallback(const json& json_param) {
    ParseParamdata(json_param);
}
// %            ... initgravity
void RegisterCallback::ParseParamdata(const nlohmann::json& json_data) {
    if (!json_data.is_object()) {
        throw std::runtime_error("JSON data must be an object");
    }
    parameter_ = json_data;
    try {
        if (!json_data.contains("register_parameter") || !json_data["register_parameter"].is_object()) {
            throw std::runtime_error("Missing or invalid 'register_parameter' object");
        }
        const auto& register_param = json_data["register_parameter"];
        // Parse parameters if they exist in the JSON object
        if (register_param.contains("num_threads")) {
            num_threads_ = register_param["num_threads"].get<int>();
        }
        if (register_param.contains("mapvoxelsize")) {
            mapvoxelsize_ = register_param["mapvoxelsize"].get<float>();
        }
        if (register_param.contains("registration_method")) {
            registration_method_ = register_param["registration_method"].get<std::string>();
        }
        if (register_param.contains("ndt_resolution")) {
            ndt_resolution_ = register_param["ndt_resolution"].get<float>();
        }
        if (register_param.contains("ndt_transform_epsilon")) {
            ndt_transform_epsilon_ = register_param["ndt_transform_epsilon"].get<float>();
        }
        if (register_param.contains("ndt_neighborhood_search_method")) {
            ndt_neighborhood_search_method_ = register_param["ndt_neighborhood_search_method"].get<std::string>();
        }
        if (register_param.contains("gicp_corr_dist_threshold")) {
            gicp_corr_dist_threshold_ = register_param["gicp_corr_dist_threshold"].get<float>();
        }
        if (register_param.contains("gicp_transform_epsilon")) {
            gicp_transform_epsilon_ = register_param["gicp_transform_epsilon"].get<float>();
        }
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("JSON parsing error in register_parameter: " + std::string(e.what()));
    }
}
// %            ... initgravity
Eigen::Vector3d RegisterCallback::lla2ned(double lat, double lon, double alt, double rlat, double rlon, double ralt) {
    // Constants according to WGS84
    constexpr double a = 6378137.0;              // Semi-major axis (m)
    constexpr double e2 = 0.00669437999014132;   // Squared eccentricity
    double dphi = lat - rlat;
    double dlam = SymmetricalAngle(lon - rlon);
    double dh = alt - ralt;
    double cp = std::cos(rlat);
    double sp = std::sin(rlat); // Fixed: was sin(originlon)
    double tmp1 = std::sqrt(1 - e2 * sp * sp);
    double tmp3 = tmp1 * tmp1 * tmp1;
    double dlam2 = dlam * dlam;   // Fixed: was dlam.*dlam
    double dphi2 = dphi * dphi;   // Fixed: was dphi.*dphi
    double E = (a / tmp1 + ralt) * cp * dlam -
            (a * (1 - e2) / tmp3 + ralt) * sp * dphi * dlam + // Fixed: was dphi.*dlam
            cp * dlam * dh;                                       // Fixed: was dlam.*dh
    double N = (a * (1 - e2) / tmp3 + ralt) * dphi +
            1.5 * cp * sp * a * e2 * dphi2 +
            sp * sp * dh * dphi +                              // Fixed: was dh.*dphi
            0.5 * sp * cp * (a / tmp1 + ralt) * dlam2;
    double D = -(dh - 0.5 * (a - 1.5 * a * e2 * cp * cp + 0.5 * a * e2 + ralt) * dphi2 -
                0.5 * cp * cp * (a / tmp1 - ralt) * dlam2);
    return Eigen::Vector3d(N, E, D);
}
// %            ... inittimestamp
Eigen::Vector3d RegisterCallback::ned2lla(double n, double e, double d, double rlat, double rlon, double ralt) {
    // Constants and spheroid properties (WGS84)
    const double a = 6378137.0; // Semi-major axis (m)
    const double f = 1.0 / 298.257223563; // Flattening
    const double b = (1.0 - f) * a; // Semi-minor axis (m)
    const double e2 = f * (2.0 - f); // Square of first eccentricity
    const double ep2 = e2 / (1.0 - e2); // Square of second eccentricity
    double slat = std::sin(rlat);
    double clat = std::cos(rlat);
    double slon = std::sin(rlon);
    double clon = std::cos(rlon);
    double Nval = a / std::sqrt(1.0 - e2 * slat * slat);
    double rho = (Nval + ralt) * clat;
    double z0 = (Nval * (1.0 - e2) + ralt) * slat;
    double x0 = rho * clon;
    double y0 = rho * slon;
    double t = clat * (-d) - slat * n;
    double dz = slat * (-d) + clat * n;
    double dx = clon * t - slon * e;
    double dy = slon * t + clon * e;
    double x = x0 + dx;
    double y = y0 + dy;
    double z = z0 + dz;
    double lon = std::atan2(y, x);
    rho = std::hypot(x, y);
    double beta = std::atan2(z, (1.0 - f) * rho);
    double lat = std::atan2(z + b * ep2 * std::pow(std::sin(beta), 3),
                            rho - a * e2 * std::pow(std::cos(beta), 3));
    double betaNew = std::atan2((1.0 - f) * std::sin(lat), std::cos(lat));
    int count = 0;
    const int maxIterations = 5;
    while (std::abs(beta - betaNew) > 1e-10 && count < maxIterations) {
        beta = betaNew;
        lat = std::atan2(z + b * ep2 * std::pow(std::sin(beta), 3),
                         rho - a * e2 * std::pow(std::cos(beta), 3));
        betaNew = std::atan2((1.0 - f) * std::sin(lat), std::cos(lat));
        count++;
    }
    slat = std::sin(lat);
    Nval = a / std::sqrt(1.0 - e2 * slat * slat);
    double alt = rho * std::cos(lat) + (z + e2 * Nval * slat) * slat - Nval;
    return Eigen::Vector3d(lat, lon, alt);
}
// %            ... initgravity
double RegisterCallback::SymmetricalAngle(double x) {
    constexpr double PI = M_PI;
    constexpr double TWO_PI = 2.0 * M_PI;
    double y = std::remainder(x, TWO_PI);
    if (y == PI) {y = -PI;}
    return y;
}
// %            ... reorderCovarianceForGTSAM
Eigen::Matrix<double, 6, 6> RegisterCallback::reorderCovarianceForGTSAM(
    const Eigen::Matrix<double, 6, 6>& ndt_covariance)
{
    Eigen::Matrix<double, 6, 6> gtsam_covariance;

    // Copy C_tt (translation-translation) from top-left to bottom-right
    gtsam_covariance.block<3, 3>(3, 3) = ndt_covariance.block<3, 3>(0, 0);

    // Copy C_rr (rotation-rotation) from bottom-right to top-left
    gtsam_covariance.block<3, 3>(0, 0) = ndt_covariance.block<3, 3>(3, 3);

    // Copy cross-correlation blocks
    gtsam_covariance.block<3, 3>(0, 3) = ndt_covariance.block<3, 3>(0, 3);
    gtsam_covariance.block<3, 3>(3, 0) = ndt_covariance.block<3, 3>(3, 0);

    return gtsam_covariance;
}

Eigen::Matrix3f RegisterCallback::Cb2n(const Eigen::Quaternionf& q) {
    // Extract quaternion components. Eigen's convention is (w, x, y, z).
    float w = q.w();
    float x = q.x();
    float y = q.y();
    float z = q.z();

    // Pre-calculate squared terms and products
    float q0q0 = w * w;
    float q1q1 = x * x;
    float q2q2 = y * y;
    float q3q3 = z * z;
    float q1q2 = x * y;
    float q0q3 = w * z;
    float q1q3 = x * z;
    float q0q2 = w * y;
    float q2q3 = y * z;
    float q0q1 = w * x;

    Eigen::Matrix3f C;
    
    // Assemble the matrix using the same formula as the MATLAB script
    C(0, 0) = q0q0 + q1q1 - q2q2 - q3q3;
    C(0, 1) = 2.0f * (q1q2 - q0q3);
    C(0, 2) = 2.0f * (q1q3 + q0q2);

    C(1, 0) = 2.0f * (q1q2 + q0q3);
    C(1, 1) = q0q0 - q1q1 + q2q2 - q3q3;
    C(1, 2) = 2.0f * (q2q3 - q0q1);

    C(2, 0) = 2.0f * (q1q3 - q0q2);
    C(2, 1) = 2.0f * (q2q3 + q0q1);
    C(2, 2) = q0q0 - q1q1 - q2q2 + q3q3;
            
    return C;
}

