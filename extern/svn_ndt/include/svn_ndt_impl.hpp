#ifndef SVN_NDT_SVN_NDT_IMPL_HPP_
#define SVN_NDT_SVN_NDT_IMPL_HPP_

// Include the class header
#include <svn_ndt.h> // Make sure this path is correct

// --- Standard/External Library Includes ---
#include <vector>
#include <cmath>
#include <numeric> // For std::accumulate
#include <limits>
#include <iostream> // For PCL_WARN/ERROR and debug output
#include <chrono>   // For timing
#include <iomanip> // For std::fixed, std::setprecision in debug output

// --- ADDED: OpenMP Fallback Functions (like in ndt_omp_impl.hpp) ---
// If OpenMP is not enabled by the compiler, _OPENMP will not be defined.
// We provide dummy functions that return 1 (for max threads) and 0 (for thread num)
// so the code compiles and runs in serial (single-threaded) mode.
#ifndef _OPENMP
#include <omp.h> // Include omp.h even if not enabled, for the stubs
int omp_get_max_threads()
{
    return 1;
}
int omp_get_thread_num()
{
    return 0;
}
#endif
// --- End of ADDED block ---


// Eigen for linear algebra
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Cholesky> // For LDLT solver
#include <Eigen/SVD>      // For condition number calculation

// PCL for point cloud operations
#include <pcl/common/transforms.h> // For pcl::transformPointCloud
#include <pcl/common/point_tests.h> // For pcl::isFinite
#include <pcl/io/pcd_io.h> // Include for PCL_WARN_STREAM

// GTSAM for pose representation and operations
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/numericalDerivative.h>

// For noise models if sampling prior
#include <gtsam/linear/Sampler.h>
#include <gtsam/linear/NoiseModel.h>


namespace svn_ndt
{

//=================================================================================================
// Constructor & Helper
//=================================================================================================
template <typename PointSource, typename PointTarget>
SvnNormalDistributionsTransform<PointSource, PointTarget>::SvnNormalDistributionsTransform()
    : target_cells_(), resolution_(1.0f), outlier_ratio_(0.55),
      search_method_(NeighborSearchMethod::DIRECT7), // Default search method to DIRECT7
      num_threads_(omp_get_max_threads()), // Initialize thread count using OpenMP
      K_(30), max_iter_(50), kernel_h_(1.0),
      // *****************************************************************
      // *** FIX 1: Changed default step size from 0.1 to 1.0 ***
      // *****************************************************************
      step_size_(1.0), 
      stop_thresh_(1e-4),
      use_gauss_newton_hessian_(true) // Default to true
{
    updateNdtConstants(); // Initialize gauss_d* constants based on defaults
    // Zero out precomputed matrices initially
    j_ang_.setZero();
    h_ang_.setZero();
}

template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::updateNdtConstants()
{
    // Recalculate NDT constants based on current resolution and outlier ratio
    // Based on NDT paper (Magnusson 2009, Eq 6.8)
    if (resolution_ <= 1e-6f) { // Use epsilon for float comparison
        PCL_ERROR("[SvnNdt] Resolution must be positive. Cannot update NDT constants.\n");
        gauss_d1_ = 1.0; gauss_d2_ = 1.0; gauss_d3_ = 0.0;
        return;
    }

    double gauss_c1 = 10.0 * (1.0 - outlier_ratio_);
    double gauss_c2 = outlier_ratio_ / pow(static_cast<double>(resolution_), 3); // Use double pow

    constexpr double epsilon = 1e-9;
    // Prevent log(0) issues
    if (gauss_c1 <= epsilon) gauss_c1 = epsilon;
    if (gauss_c2 <= epsilon) gauss_c2 = epsilon;

    double c1_plus_c2 = gauss_c1 + gauss_c2;
    gauss_d3_ = -log(gauss_c2);
    gauss_d1_ = -log(c1_plus_c2) - gauss_d3_;

    // Calculate gauss_d2 directly using the formula from Magnusson Eq 6.8
    // Note: Ensure gauss_d1_ is not zero before dividing.
    if (std::abs(gauss_d1_) < epsilon) {
        PCL_ERROR("[SvnNdt] gauss_d1_ is near zero during gauss_d2_ calculation. Check outlier_ratio/resolution. Setting gauss_d2_ to default 1.0.\n");
        gauss_d2_ = 1.0;
    } else {
        // Calculate the argument for the inner log
        double inner_log_arg = gauss_c1 * exp(-0.5) + gauss_c2;
        if (inner_log_arg <= epsilon) {
             PCL_WARN("[SvnNdt] Inner log argument for gauss_d2_ is near zero/negative (%.3e). Setting gauss_d2_ to default 1.0.\n", inner_log_arg);
             gauss_d2_ = 1.0;
        } else {
             // Calculate the argument for the outer log
             double d2_outer_log_arg = (-log(inner_log_arg) - gauss_d3_) / gauss_d1_;
             if (d2_outer_log_arg <= epsilon) {
                  PCL_WARN("[SvnNdt] Outer log argument for gauss_d2_ is near zero/negative (%.3e). Setting gauss_d2_ to default 1.0.\n", d2_outer_log_arg);
                  gauss_d2_ = 1.0;
             } else {
                  // Final calculation
                  gauss_d2_ = -2.0 * log(d2_outer_log_arg);
             }
        }
    }

    if (!std::isfinite(gauss_d1_) || !std::isfinite(gauss_d2_) || !std::isfinite(gauss_d3_)) {
        PCL_ERROR("[SvnNdt] NaN/Inf detected in NDT constant calculation. Check resolution (%.3f) and outlier ratio (%.3f).\n", resolution_, outlier_ratio_);
        // Provide some default values to prevent further issues
        gauss_d1_ = 1.0; gauss_d2_ = 1.0; gauss_d3_ = 0.0;
    }
}

//=================================================================================================
// Configuration
//=================================================================================================
template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::setInputTarget(
    const PointCloudTargetConstPtr& cloud)
{
    if (!cloud || cloud->empty()) {
        PCL_ERROR("[SvnNdt::setInputTarget] Invalid or empty target point cloud provided.\n");
        target_cells_.setInputCloud(nullptr); // Clear previous target if any
        return;
    }
    target_cells_.setInputCloud(cloud);

    if (resolution_ > 1e-6f) {
        target_cells_.setLeafSize(resolution_, resolution_, resolution_);
        bool build_kdtree = (search_method_ == NeighborSearchMethod::KDTREE);
        target_cells_.filter(build_kdtree); // Builds the voxel grid
        updateNdtConstants(); // Ensure constants are up-to-date
        PCL_INFO("[SvnNdt::setInputTarget] Built NDT target grid with %zu valid voxels.\n", target_cells_.getAllLeaves().size());
        if (target_cells_.getAllLeaves().empty()){
             PCL_WARN("[SvnNdt::setInputTarget] Warning: Target NDT grid contains zero valid voxels. Check resolution or min_points_per_voxel.\n");
        }
    } else {
        PCL_WARN("[SvnNdt::setInputTarget] Target cloud set, but resolution is not positive. Grid not built.\n");
    }
}

template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::setResolution(float resolution)
{
    if (resolution <= 1e-6f) {
        PCL_ERROR("[SvnNdt::setResolution] Resolution must be positive.\n");
        return;
    }
    // Only rebuild if resolution actually changed significantly
    if (std::abs(resolution_ - resolution) > 1e-6f) {
        resolution_ = resolution;
        // Rebuild the NDT map if a target cloud is already loaded
        if (target_cells_.getInputCloud()) {
            setInputTarget(target_cells_.getInputCloud());
        }
    }
}

template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::setMinPointPerVoxel(int min_points)
{
    // Pass the setting directly to the internal voxel grid object
    target_cells_.setMinPointPerVoxel(min_points);
    // Rebuild the NDT map if a target cloud is already loaded, as this changes voxel validity
    if (target_cells_.getInputCloud()) {
        setInputTarget(target_cells_.getInputCloud());
    }
}


template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::setOutlierRatio(double ratio)
{
    double clamped_ratio = ratio;
    if (ratio < 0.0) {
         PCL_WARN("[SvnNdt::setOutlierRatio] Outlier ratio must be non-negative. Clamping to 0.0.\n");
         clamped_ratio = 0.0;
    } else if (ratio >= 1.0) {
         // Allow ratio = 1.0? Magnusson uses 1-ratio, so 1.0 is problematic. Clamp slightly below.
         PCL_WARN("[SvnNdt::setOutlierRatio] Outlier ratio must be less than 1.0. Clamping near 1.0.\n");
         clamped_ratio = 1.0 - 1e-9;
    }

    if (std::abs(outlier_ratio_ - clamped_ratio) > 1e-9) {
        outlier_ratio_ = clamped_ratio;
        updateNdtConstants(); // Recalculate gauss_d* constants
    }
}


//=================================================================================================
// RBF Kernel Functions (Implementations for SE(3))
//=================================================================================================
template <typename PointSource, typename PointTarget>
double SvnNormalDistributionsTransform<PointSource, PointTarget>::rbf_kernel(
    const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const
{
    // RBF kernel: k(x, y) = exp(-||Log(x^{-1}y)||^2 / h)
    if (kernel_h_ <= 1e-12) { // Avoid division by zero
        return (pose_l.equals(pose_k, 1e-9)) ? 1.0 : 0.0; // Delta function approx
    }
    // Use Logmap(pose_l.between(pose_k)) which gives tangent vector at pose_l
    Vector6d diff_log = gtsam::Pose3::Logmap(pose_l.between(pose_k));
    double sq_norm = diff_log.squaredNorm();
    return std::exp(-sq_norm / kernel_h_);
}

template <typename PointSource, typename PointTarget>
typename SvnNormalDistributionsTransform<PointSource, PointTarget>::Vector6d
SvnNormalDistributionsTransform<PointSource, PointTarget>::rbf_kernel_gradient(
    const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const
{
     // Gradient of k(l, k) w.r.t. pose_l
     if (kernel_h_ <= 1e-12) {
         return Vector6d::Zero(); // Gradient of delta function approx is zero everywhere except origin
     }
    // Logmap(T_l^{-1} * T_k) -> tangent vector xi at T_l
    Vector6d diff_log = gtsam::Pose3::Logmap(pose_l.between(pose_k));
    double sq_norm = diff_log.squaredNorm();
    double k_val = std::exp(-sq_norm / kernel_h_);

    // Derivative: exp(-xi^2/h) * (-2/h) * xi
    // Result is a tangent vector at pose_l
    return k_val * (-2.0 / kernel_h_) * diff_log;
}


//=================================================================================================
// NDT Math Functions (Adapted SERIAL Implementations)
// These functions assume the NDT standard [x,y,z,roll,pitch,yaw] pose vector convention.
// This is OK, as long as we permute the final output before mixing with GTSAM.
//=================================================================================================

// --- computeAngleDerivatives ---
template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::computeAngleDerivatives(
    const Vector6d& p, bool compute_hessian) // p is expected [x,y,z,r,p,y]
{
    // Use roll, pitch, yaw from the input vector p
    // NDT uses x-y-z Tait-Bryan angles: p(3)=roll, p(4)=pitch, p(5)=yaw
    double r = p(3), pi = p(4), y = p(5);

    // Use double precision for trig functions
    double cx, cy, cz, sx, sy, sz;
    constexpr double angle_epsilon = 1e-7; // Use a slightly smaller epsilon for double
    if (std::abs(r) < angle_epsilon) { sx = 0.0; cx = 1.0; } else { sx = sin(r); cx = cos(r); }
    if (std::abs(pi) < angle_epsilon) { sy = 0.0; cy = 1.0; } else { sy = sin(pi); cy = cos(pi); }
    if (std::abs(y) < angle_epsilon) { sz = 0.0; cz = 1.0; } else { sz = sin(y); cz = cos(y); }

    // --- Jacobian Components (Magnusson Eq 6.19) ---
    // Calculate intermediate vectors (using double)
    Eigen::Vector3d j_ang_a_d, j_ang_b_d, j_ang_c_d, j_ang_d_d, j_ang_e_d, j_ang_f_d, j_ang_g_d, j_ang_h_d;
    j_ang_a_d << (-sx*sz+cx*sy*cz), (-sx*cz-cx*sy*sz), (-cx*cy);
    j_ang_b_d << ( cx*sz+sx*sy*cz), ( cx*cz-sx*sy*sz), (-sx*cy);
    j_ang_c_d << (-sy*cz),          ( sy*sz),          ( cy);
    j_ang_d_d << ( sx*cy*cz),       (-sx*cy*sz),       ( sx*sy);
    j_ang_e_d << (-cx*cy*cz),       ( cx*cy*sz),       (-cx*sy);
    j_ang_f_d << (-cy*sz),          (-cy*cz),          ( 0.0);
    j_ang_g_d << ( cx*cz-sx*sy*sz), (-cx*sz-sx*sy*cz), ( 0.0);
    j_ang_h_d << ( sx*cz+cx*sy*sz), ( cx*sy*cz-sx*sz), ( 0.0);

    // Assign to the member matrix (float)
    j_ang_.setZero();
    j_ang_.row(0).head<3>() = j_ang_a_d.cast<float>(); // dR'/dr
    j_ang_.row(1).head<3>() = j_ang_b_d.cast<float>(); // dR'/dr
    j_ang_.row(2).head<3>() = j_ang_c_d.cast<float>(); // dR'/dp
    j_ang_.row(3).head<3>() = j_ang_d_d.cast<float>(); // dR'/dp
    j_ang_.row(4).head<3>() = j_ang_e_d.cast<float>(); // dR'/dp
    j_ang_.row(5).head<3>() = j_ang_f_d.cast<float>(); // dR'/dy
    j_ang_.row(6).head<3>() = j_ang_g_d.cast<float>(); // dR'/dy
    j_ang_.row(7).head<3>() = j_ang_h_d.cast<float>(); // dR'/dy

    // --- Hessian Components (Magnusson Eq 6.21) ---
    if (compute_hessian) {
        // Calculate intermediate vectors (using double)
        Eigen::Vector3d h_ang_a2_d, h_ang_a3_d, h_ang_b2_d, h_ang_b3_d, h_ang_c2_d, h_ang_c3_d;
        Eigen::Vector3d h_ang_d1_d, h_ang_d2_d, h_ang_d3_d, h_ang_e1_d, h_ang_e2_d, h_ang_e3_d;
        Eigen::Vector3d h_ang_f1_d, h_ang_f2_d, h_ang_f3_d;

        h_ang_a2_d << (-cx*sz - sx*sy*cz), (-cx*cz + sx*sy*sz), ( sx*cy); // H_rr(y)
        h_ang_a3_d << (-sx*sz + cx*sy*cz), (-cx*sy*sz - sx*cz), (-cx*cy); // H_rr(z)
        h_ang_b2_d << ( cx*cy*cz),         (-cx*cy*sz),         ( cx*sy); // H_rp(y)
        h_ang_b3_d << ( sx*cy*cz),         (-sx*cy*sz),         ( sx*sy); // H_rp(z)
        h_ang_c2_d << (-sx*cz - cx*sy*sz), ( sx*sz - cx*sy*cz), ( 0.0);   // H_ry(y)
        h_ang_c3_d << ( cx*cz - sx*sy*sz), (-sx*sy*cz - cx*sz), ( 0.0);   // H_ry(z)
        h_ang_d1_d << (-cy*cz),            ( cy*sz),            ( sy);    // H_pp(x)
        h_ang_d2_d << (-sx*sy*cz),         ( sx*sy*sz),         ( sx*cy); // H_pp(y)
        h_ang_d3_d << ( cx*sy*cz),         (-cx*sy*sz),         (-cx*cy); // H_pp(z)
        h_ang_e1_d << ( sy*sz),            ( sy*cz),            ( 0.0);   // H_py(x)
        h_ang_e2_d << (-sx*cy*sz),         (-sx*cy*cz),         ( 0.0);   // H_py(y)
        h_ang_e3_d << ( cx*cy*sz),         ( cx*cy*cz),         ( 0.0);   // H_py(z)
        h_ang_f1_d << (-cy*cz),            ( cy*sz),            ( 0.0);   // H_yy(x) = H_py(x)
        h_ang_f2_d << (-cx*sz - sx*sy*cz), (-cx*cz + sx*sy*sz), ( 0.0);   // H_yy(y)
        h_ang_f3_d << (-sx*sz + cx*sy*cz), (-cx*sy*sz - sx*cz), ( 0.0);   // H_yy(z)

        // Assign to the member matrix (float)
        h_ang_.setZero();
        h_ang_.row(0).head<3>() = h_ang_a2_d.cast<float>(); // H_rr(y)
        h_ang_.row(1).head<3>() = h_ang_a3_d.cast<float>(); // H_rr(z)
        h_ang_.row(2).head<3>() = h_ang_b2_d.cast<float>(); // H_rp(y)
        h_ang_.row(3).head<3>() = h_ang_b3_d.cast<float>(); // H_rp(z)
        h_ang_.row(4).head<3>() = h_ang_c2_d.cast<float>(); // H_ry(y)
        h_ang_.row(5).head<3>() = h_ang_c3_d.cast<float>(); // H_ry(z)
        h_ang_.row(6).head<3>() = h_ang_d1_d.cast<float>(); // H_pp(x)
        h_ang_.row(7).head<3>() = h_ang_d2_d.cast<float>(); // H_pp(y)
        h_ang_.row(8).head<3>() = h_ang_d3_d.cast<float>(); // H_pp(z)
        h_ang_.row(9).head<3>() = h_ang_e1_d.cast<float>(); // H_py(x)
        h_ang_.row(10).head<3>() = h_ang_e2_d.cast<float>(); // H_py(y)
        h_ang_.row(11).head<3>() = h_ang_e3_d.cast<float>(); // H_py(z)
        h_ang_.row(12).head<3>() = h_ang_f1_d.cast<float>(); // H_yy(x)
        h_ang_.row(13).head<3>() = h_ang_f2_d.cast<float>(); // H_yy(y)
        h_ang_.row(14).head<3>() = h_ang_f3_d.cast<float>(); // H_yy(z)
        // Row 15 remains zero
    }
}


// --- computePointDerivatives ---
template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::computePointDerivatives(
    const Eigen::Vector3d& x,                     // Original point coordinates
    Eigen::Matrix<float, 4, 6>& point_gradient_, // Output Jacobian Jp (4x6)
    Eigen::Matrix<float, 24, 6>& point_hessian_, // Output Hessian Hp (flattened 24x6)
    bool compute_hessian) const // <<< ADDED CONST
{
    // Point in homogeneous coordinates (as float)
    Eigen::Vector4f x4(static_cast<float>(x[0]), static_cast<float>(x[1]), static_cast<float>(x[2]), 0.0f);

    // --- Jacobian Calculation (Magnusson Eq 6.18) ---
    // --- NOTE: Translation part (Identity) is set by the CALLING function ---
    // --- This function ONLY fills in the angular components (cols 3, 4, 5) ---

    // Calculate effect of angular derivatives on this point using precomputed j_ang_
    Eigen::Matrix<float, 8, 1> x_j_ang = j_ang_ * x4; // (8x4) * (4x1) -> (8x1)

    // Assign derivatives w.r.t. rotation (columns 3, 4, 5)
    point_gradient_(1, 3) = x_j_ang[0]; // y' w.r.t roll
    point_gradient_(2, 3) = x_j_ang[1]; // z' w.r.t roll
    point_gradient_(0, 4) = x_j_ang[2]; // x' w.r.t pitch
    point_gradient_(1, 4) = x_j_ang[3]; // y' w.r.t pitch
    point_gradient_(2, 4) = x_j_ang[4]; // z' w.r.t pitch
    point_gradient_(0, 5) = x_j_ang[5]; // x' w.r.t yaw
    point_gradient_(1, 5) = x_j_ang[6]; // y' w.r.t yaw
    point_gradient_(2, 5) = x_j_ang[7]; // z' w.r.t yaw

    // --- Hessian Calculation (Magnusson Eq 6.20 & 6.21) ---
    if (compute_hessian) {
        point_hessian_.setZero(); // Initialize output Hessian
        // Calculate effect of angular second derivatives using precomputed h_ang_
        Eigen::Matrix<float, 16, 1> x_h_ang = h_ang_ * x4; // (16x4) * (4x1) -> (16x1)

        // Assign second derivatives w.r.t. rotation pairs
        // Indices map directly from Magnusson Eq 6.21 components to flattened structure
        point_hessian_(13, 3) = x_h_ang[0];  // H_rr(y)
        point_hessian_(14, 3) = x_h_ang[1];  // H_rr(z)
        point_hessian_(13, 4) = x_h_ang[2];  // H_rp(y)
        point_hessian_(14, 4) = x_h_ang[3];  // H_rp(z)
        point_hessian_(17, 3) = x_h_ang[2];  // H_pr(y) = H_rp(y)
        point_hessian_(18, 3) = x_h_ang[3];  // H_pr(z) = H_rp(z)
        point_hessian_(13, 5) = x_h_ang[4];  // H_ry(y)
        point_hessian_(14, 5) = x_h_ang[5];  // H_ry(z)
        point_hessian_(21, 3) = x_h_ang[4];  // H_yr(y) = H_ry(y)
        point_hessian_(22, 3) = x_h_ang[5];  // H_yr(z) = H_ry(z)
        point_hessian_(16, 4) = x_h_ang[6];  // H_pp(x)
        point_hessian_(17, 4) = x_h_ang[7];  // H_pp(y)
        point_hessian_(18, 4) = x_h_ang[8];  // H_pp(z)
        point_hessian_(16, 5) = x_h_ang[9];  // H_py(x)
        point_hessian_(17, 5) = x_h_ang[10]; // H_py(y)
        point_hessian_(18, 5) = x_h_ang[11]; // H_py(z)
        point_hessian_(20, 4) = x_h_ang[9];  // H_yp(x) = H_py(x)
        point_hessian_(21, 4) = x_h_ang[10]; // H_yp(y) = H_py(y)
        point_hessian_(22, 4) = x_h_ang[11]; // H_yp(z) = H_py(z)
        point_hessian_(20, 5) = x_h_ang[12]; // H_yy(x)
        point_hessian_(21, 5) = x_h_ang[13]; // H_yy(y)
        point_hessian_(22, 5) = x_h_ang[14]; // H_yy(z)
    }
}


// --- updateDerivatives ---
template <typename PointSource, typename PointTarget>
double SvnNormalDistributionsTransform<PointSource, PointTarget>::updateDerivatives(
    Vector6d& score_gradient, // Accumulated Gradient (output)
    Matrix6d& hessian,        // Accumulated Hessian (output)
    const Eigen::Matrix<float, 4, 6>& point_gradient4, // Jacobian of point transform w.r.t p=[x,y,z,r,p,y]
    const Eigen::Matrix<float, 24, 6>& point_hessian_, // Hessian of point transform w.r.t p=[x,y,z,r,p,y]
    const Eigen::Vector3d& x_trans, // Point relative to voxel mean (point - mean)
    const Eigen::Matrix3d& c_inv,   // Voxel inverse covariance
    bool compute_hessian,
    bool use_gauss_newton_hessian) const // <<< ADDED CONST
{
    // Homogeneous vector of point relative to mean (use float for consistency with derivatives)
    Eigen::Matrix<float, 1, 4> x_trans4(static_cast<float>(x_trans[0]), static_cast<float>(x_trans[1]), static_cast<float>(x_trans[2]), 0.0f);
    // Inverse covariance padded to 4x4 (as float)
    Eigen::Matrix4f c_inv4 = Eigen::Matrix4f::Zero();
    c_inv4.topLeftCorner(3, 3) = c_inv.cast<float>();

    // Calculate Mahalanobis distance squared: (x'-mu)^T * Sigma^-1 * (x'-mu)
    double mahal_sq = x_trans.dot(c_inv * x_trans); // Use double for precision

    // --- Safety Checks for Score Calculation ---
    constexpr double max_exponent_arg = 50.0; // Prevent exp() overflow/underflow
    if (!std::isfinite(mahal_sq) || mahal_sq < -1e-9 ) { // Allow small negative due to precision
        // PCL_DEBUG("[SvnNdt::updateDeriv] Invalid Mahalanobis dist^2: %.3e\n", mahal_sq); // Use DEBUG
        return 0.0; // Return zero score contribution
    }
    // Clamp small negative values resulting from numerical precision issues
    if (mahal_sq < 0.0) mahal_sq = 0.0;

    // Avoid large exponents
    double exponent_arg = gauss_d2_ * mahal_sq * 0.5;
    if (exponent_arg > max_exponent_arg) {
        // PCL_DEBUG("[SvnNdt::updateDeriv] Exponent argument too large: %.3e\n", exponent_arg); // Use DEBUG
        return 0.0; // exp() would be huge/inf, score contribution negligible
    }

    // Calculate exponential term (Eq 6.9)
    double exp_term = std::exp(-exponent_arg);
    // Calculate score increment (Eq 6.9)
    double score_inc = -gauss_d1_ * exp_term;

    // --- Safety Check for Gradient/Hessian Factor ---
    // This factor multiplies all gradient and Hessian terms
    double factor = gauss_d1_ * gauss_d2_ * exp_term;
    if (!std::isfinite(factor) || std::abs(factor) < 1e-15) { // Check for NaN/Inf or near-zero
         // PCL_DEBUG("[SvnNdt::updateDeriv] Factor is invalid or near zero: %.3e\n", factor); // Use DEBUG
        return score_inc; // Return score but don't update derivatives
    }

    // --- Gradient Calculation (Eq 6.12) ---
    // temp_vec = C^-1 * Jp (4x6)
    Eigen::Matrix<float, 4, 6> temp_vec = c_inv4 * point_gradient4;
    // grad_contrib_float = (x-mu)^T * C^-1 * Jp (1x6)
    Eigen::Matrix<float, 1, 6> grad_contrib_float = x_trans4 * temp_vec;
    // grad_inc = factor * [ (x-mu)^T * C^-1 * Jp ]^T
    Vector6d grad_inc = factor * grad_contrib_float.transpose().cast<double>();

    // Accumulate gradient if valid
    if (grad_inc.allFinite()) {
        score_gradient += grad_inc;
    } else {
        PCL_WARN("[SvnNdt::updateDeriv] NaN/Inf in gradient increment calculation.\n");
    }

    // --- Hessian Calculation (Eq 6.13 or Gauss-Newton Approx) ---
    if (compute_hessian) {
        Matrix6d hess_contrib = Matrix6d::Zero();

        // Term 2 (Gauss-Newton): Jp^T * C^-1 * Jp (ALWAYS included)
        // Note: temp_vec = C^-1 * Jp
        Matrix6d term2 = (point_gradient4.transpose() * temp_vec).cast<double>();

        if (!use_gauss_newton_hessian) {
            // Calculate terms needed only for Full Analytical Hessian
            Eigen::Matrix<double, 1, 6> grad_contrib_double = grad_contrib_float.cast<double>();

            // Term 1: -d2 * [grad_contrib]^T * [grad_contrib]
            Matrix6d term1 = -gauss_d2_ * (grad_contrib_double.transpose() * grad_contrib_double);

            // Term 3: (x-mu)^T * C^-1 * Hp
            Eigen::Matrix<float, 1, 4> x_trans4_c_inv4 = x_trans4 * c_inv4;
            Matrix6d term3 = Matrix6d::Zero();
            for (int i = 0; i < 6; ++i) { // Row index of final Hessian H_ij
                for (int j = i; j < 6; ++j) { // Col index of final Hessian H_ij
                     // Extract the 4x1 block for d^2(x')/(dp_i dp_j)
                     Eigen::Matrix<float, 4, 1> H_ij_x = point_hessian_.block<4, 1>(i * 4, j);
                     // Calculate (x-mu)^T * C^-1 * [d^2(x')/(dp_i dp_j)]
                     term3(i, j) = x_trans4_c_inv4 * H_ij_x;
                }
            }
            // Exploit symmetry H_ij = H_ji
            term3.template triangularView<Eigen::Lower>() = term3.template triangularView<Eigen::Upper>().transpose();

            // Combine terms for Full Analytical Hessian
            hess_contrib = term1 + term2 + term3;

        } else {
            // Use only Gauss-Newton term
            hess_contrib = term2;
        }

        // Scale total contribution by factor
        hess_contrib *= factor;

        // Accumulate Hessian if valid
        if (hess_contrib.allFinite()){
            hessian += hess_contrib;
        } else {
             PCL_WARN("[SvnNdt::updateDeriv] NaN/Inf in Hessian increment calculation (GN=%d).\n", use_gauss_newton_hessian);
        }
    } // End if compute_hessian

    return score_inc; // Return the score contribution
}


// --- computeParticleDerivatives (Now SERIAL, to be called in parallel) ---
template <typename PointSource, typename PointTarget>
double SvnNormalDistributionsTransform<PointSource, PointTarget>::computeParticleDerivatives(
    Vector6d& final_score_gradient, // Output gradient [x,y,z,r,p,y]
    Matrix6d& final_hessian,        // Output Hessian [x,y,z,r,p,y]
    const PointCloudSource& trans_cloud, // Transformed source cloud
    const Vector6d& p,        // Current pose estimate [x,y,z,r,p,y]
    bool compute_hessian) // Flag remains, but controls GN Hessian calculation
{
    // --- Precompute Angle Derivatives (Done once per particle) ---
    // These populate the member variables j_ang_ and h_ang_
    computeAngleDerivatives(p, compute_hessian);

    // --- Allocate Per-Point Accumulator Vectors (like pclomp) ---
    const size_t num_points = trans_cloud.points.size();
    if (num_points == 0) {
        PCL_WARN("[SvnNdt::computeParticleDerivatives] Transformed cloud is empty.\n");
        final_score_gradient.setZero();
        final_hessian.setZero();
        return 0.0;
    }

    std::vector<double> scores(num_points, 0.0);
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> score_gradients(num_points, Vector6d::Zero());
    std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>> hessians(num_points, Matrix6d::Zero());


    // *****************************************************************
    // *** FIX 2: Removed thread-local buffer pre-allocation ***
    // *****************************************************************


    // *****************************************************************
    // *** FIX 2: Removed OpenMP pragma from this inner loop ***
    // *****************************************************************
    for (size_t idx = 0; idx < num_points; ++idx) {

        // --- Thread-local variables for calculations within this loop ---
        Eigen::Matrix<float, 4, 6> point_gradient4;
        Eigen::Matrix<float, 24, 6> point_hessian24; // Only computed if flag is true
        
        // *****************************************************************
        // *** FIX 2: Declare neighbor buffers locally inside the loop ***
        // *****************************************************************
        std::vector<LeafConstPtr> neighborhood;
        std::vector<float> distances;
        constexpr size_t reserve_size = 27; // Max neighbors for DIRECT26/KDTREE radius
        neighborhood.reserve(reserve_size);
        distances.reserve(reserve_size); 

        // Access parent members needed inside the loop (const access is safe)
        const auto& target_cells = this->target_cells_;
        float resolution = this->resolution_;
        auto search_method = this->search_method_;

        // --- Get Transformed Point ---
        const PointSource& x_trans_pt = trans_cloud[idx];
        if (!pcl::isFinite(x_trans_pt)) continue; // Skip invalid points

        // --- Neighbor Search ---
        int neighbors_found = 0;
        switch (search_method) {
             case NeighborSearchMethod::KDTREE:
                 neighbors_found = target_cells.radiusSearch(x_trans_pt, resolution, neighborhood, distances);
                 break;
             // case NeighborSearchMethod::DIRECT26: // Needs VoxelGridCovariance implementation
             //     neighbors_found = target_cells.getNeighborhoodAtPoint26(x_trans_pt, neighborhood);
             //     break;
             case NeighborSearchMethod::DIRECT7:
                 neighbors_found = target_cells.getNeighborhoodAtPoint7(x_trans_pt, neighborhood);
                 break;
             case NeighborSearchMethod::DIRECT1:
             default:
                 neighbors_found = target_cells.getNeighborhoodAtPoint1(x_trans_pt, neighborhood);
                 break;
        }
        if (neighbors_found == 0) continue; // Skip points with no valid neighbors

        // --- Get Original Point ---
        if (idx >= input_->size()){
            // This should not happen if trans_cloud is from input_
            continue; 
        }
        const PointSource& x_pt = (*input_)[idx];
        Eigen::Vector3d x_orig(x_pt.x, x_pt.y, x_pt.z);

        // --- Initialize Point Gradient (Jacobian) for this point ---
        point_gradient4.setZero();
        point_gradient4.block<3, 3>(0, 0).setIdentity();

        // --- Compute Point Derivatives (fills angular part using member j_ang_, h_ang_) ---
        // This is thread-safe as it only *reads* member variables
        this->computePointDerivatives(x_orig, point_gradient4, point_hessian24, compute_hessian);

        // --- Create local accumulators for *this point* ---
        double point_score_acc = 0.0;
        Vector6d point_grad_acc = Vector6d::Zero();
        Matrix6d point_hess_acc = Matrix6d::Zero();

        // --- Accumulate Contributions from Neighbors ---
        for (const LeafConstPtr& cell : neighborhood) {
            if (!cell) continue; // Skip nullptrs

            Eigen::Vector3d x_rel = Eigen::Vector3d(x_trans_pt.x, x_trans_pt.y, x_trans_pt.z) - cell->getMean();
            const Eigen::Matrix3d& c_inv = cell->getInverseCov();

            // --- Call updateDerivatives using parent's constants (gauss_d1_, etc.) ---
            // --- Accumulate results into the *point-local* variables ---
            point_score_acc += this->updateDerivatives(point_grad_acc, point_hess_acc, // Accumulate here
                                               point_gradient4, point_hessian24,
                                               x_rel, c_inv,
                                               compute_hessian,
                                               this->use_gauss_newton_hessian_);
        } // End loop over neighbors

        // --- Write accumulated results to the per-point vectors (thread-safe) ---
        scores[idx] = point_score_acc;
        score_gradients[idx] = point_grad_acc;
        hessians[idx] = point_hess_acc;

    } // End SERIAL for loop over points


    // --- Final Serial Reduction (Summing up the per-point results) ---
    final_score_gradient.setZero();
    final_hessian.setZero();
    double final_total_score = 0.0;
    for (size_t i = 0; i < num_points; ++i) {
        final_total_score += scores[i];
        final_score_gradient += score_gradients[i];
        final_hessian += hessians[i];
    }

    // --- Hessian Regularization (Applied once per particle after reduction) ---
    if (compute_hessian) {
        constexpr double lambda = 1e-6; // Use a smaller lambda for GN
        final_hessian += lambda * Matrix6d::Identity();
    }

    // --- Final Sanity Checks (Applied once per particle) ---
    if (!final_score_gradient.allFinite()) {
        PCL_ERROR("[SvnNdt::computeParticleDerivatives] Final score_gradient contains NaN/Inf! Resetting to zero.\n");
        final_score_gradient.setZero(); // Prevent optimizer failure
    }
    if (compute_hessian && !final_hessian.allFinite()) {
        PCL_ERROR("[SvnNdt::computeParticleDerivatives] Final hessian (GN=%d) contains NaN/Inf! Resetting to identity.\n", use_gauss_newton_hessian_);
        final_hessian = Matrix6d::Identity(); // Prevent solver failure
    }

    // Return the total accumulated score for this particle
    return final_total_score;

} // End computeParticleDerivatives (Serial Version)


//=================================================================================================
// Main Alignment Function (SVN-NDT Implementation) - Using OpenMP
//=================================================================================================
template <typename PointSource, typename PointTarget>
SvnNdtResult SvnNormalDistributionsTransform<PointSource, PointTarget>::align(
    const PointCloudSource& source_cloud,
    const gtsam::Pose3& prior_mean)
{
    SvnNdtResult result; // Initialize result struct

    // --- Input Sanity Checks ---
    if (!target_cells_.getInputCloud() || target_cells_.getAllLeaves().empty()) {
        PCL_ERROR("[SvnNdt::align] Target NDT grid is not initialized. Call setInputTarget() first.\n");
        result.converged = false;
        result.final_pose = prior_mean;
        result.final_covariance.setIdentity();
        return result;
    }
    if (source_cloud.empty()) {
        PCL_ERROR("[SvnNdt::align] Input source cloud is empty.\n");
        result.converged = false;
        result.final_pose = prior_mean;
        result.final_covariance.setIdentity();
        return result;
    }
    if (K_ <= 0) {
        PCL_ERROR("[SvnNdt::align] Particle count (K_) must be positive.\n");
        result.converged = false;
        result.final_pose = prior_mean;
        result.final_covariance.setIdentity();
        return result;
    }

    // --- Initialization ---
    input_ = source_cloud.makeShared(); // Store shared ptr

    std::vector<gtsam::Pose3, Eigen::aligned_allocator<gtsam::Pose3>> particles(K_);

    // Initialize particles by sampling around the prior mean
    Vector6d initial_sigmas; initial_sigmas << 0.01, 0.01, 0.02, 0.05, 0.05, 0.05;
    auto prior_noise_model = gtsam::noiseModel::Diagonal::Sigmas(initial_sigmas);
    gtsam::Sampler sampler(prior_noise_model, std::chrono::system_clock::now().time_since_epoch().count());

    for (int k = 0; k < K_; ++k) {
        particles[k] = prior_mean.retract(sampler.sample());
    }
    
    // --- MODIFICATION: Initialize mean pose tracking ---
    gtsam::Pose3 mean_pose_current = prior_mean; // Initialize with prior
    gtsam::Pose3 mean_pose_last_iter;
    // --- END MODIFICATION ---

    // Pre-allocate vectors
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> loss_gradients(K_);
    std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>> loss_hessians(K_);
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> particle_updates(K_);
    std::vector<PointCloudSource> transformed_clouds(K_); // One per particle

    Matrix6d I6 = Matrix6d::Identity();

    // Permutation Matrix (NDT [x,y,z,r,p,y] -> GTSAM [r,p,y,x,y,z])
    Eigen::Matrix<double, 6, 6> P_gtsam_from_ndt;
    P_gtsam_from_ndt.setZero();
    P_gtsam_from_ndt.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    P_gtsam_from_ndt.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity();


    // --- SVN Iteration Loop ---
    // double avg_update_norm = std::numeric_limits<double>::max(); // No longer used for convergence
    for (int iter = 0; iter < max_iter_; ++iter)
    {
        auto iter_start_time = std::chrono::high_resolution_clock::now();
        
        // --- MODIFICATION: Store mean pose from previous iteration ---
        mean_pose_last_iter = mean_pose_current;
        // --- END MODIFICATION ---

        // --- Stage 1: Compute NDT Derivatives (Outer loop is SERIAL) ---
        auto stage1_start_time = std::chrono::high_resolution_clock::now();
        
        Vector6d grad_k_ndt; // NDT order [x,y,z,r,p,y], Gradient of Cost
        Matrix6d hess_k_ndt; // NDT order [x,y,z,r,p,y], Hessian of Cost

        // *****************************************************************
        // *** FIX 2: Added OpenMP pragma to this outer loop ***
        // *** Made grad_k_ndt and hess_k_ndt private       ***
        // *****************************************************************
        #pragma omp parallel for num_threads(num_threads_) schedule(dynamic, 1) private(grad_k_ndt, hess_k_ndt)
        for (int k = 0; k < K_; ++k) {
            // Transform point cloud using current particle pose
            pcl::transformPointCloud(source_cloud, transformed_clouds[k], particles[k].matrix().cast<float>());

            // Convert particle's gtsam::Pose3 to NDT's expected [x,y,z,r,p,y] vector
            Vector6d p_k_ndt;
            gtsam::Vector3 rpy = particles[k].rotation().rpy();
            p_k_ndt.head<3>() = particles[k].translation();
            p_k_ndt.tail<3>() = rpy;

            // --- Call the (now serial) computeParticleDerivatives ---
            double score_k = this->computeParticleDerivatives(grad_k_ndt, hess_k_ndt,
                                                               transformed_clouds[k], p_k_ndt,
                                                               true); // compute_hessian = true

            // Store results
            loss_gradients[k] = grad_k_ndt;
            loss_hessians[k] = hess_k_ndt;

             // Sanity checks remain the same
             if (!loss_gradients[k].allFinite()) { /* Warn/Reset */ loss_gradients[k].setZero(); }
             if (!loss_hessians[k].allFinite()) { /* Warn/Reset */ loss_hessians[k] = I6; }
        }
        auto stage1_end_time = std::chrono::high_resolution_clock::now();


        // --- Stage 2: Calculate SVN Updates (Parallel using OpenMP) ---
        auto stage2_start_time = std::chrono::high_resolution_clock::now();
        
        // This pragma will be *ignored* if OpenMP is not enabled.
        #pragma omp parallel for num_threads(num_threads_) schedule(dynamic)
        for (int k = 0; k < K_; ++k) {
            
            Vector6d phi_k_star_gtsam = Vector6d::Zero();
            Matrix6d H_k_tilde_gtsam = Matrix6d::Zero();

            for (int l = 0; l < K_; ++l) {
                double k_val = rbf_kernel(particles[l], particles[k]);
                Vector6d k_grad_l = rbf_kernel_gradient(particles[l], particles[k]);
                if (!std::isfinite(k_val) || !k_grad_l.allFinite()) { continue; }

                Vector6d grad_l_gtsam = P_gtsam_from_ndt * loss_gradients[l];
                Matrix6d hess_l_gtsam = P_gtsam_from_ndt * loss_hessians[l] * P_gtsam_from_ndt;

                if (grad_l_gtsam.allFinite()) { phi_k_star_gtsam += k_val * grad_l_gtsam; }
                phi_k_star_gtsam += k_grad_l;

                if (hess_l_gtsam.allFinite()) { H_k_tilde_gtsam += (k_val * k_val) * hess_l_gtsam; }
                H_k_tilde_gtsam += (k_grad_l * k_grad_l.transpose());
            }

            if (K_ > 0) {
                phi_k_star_gtsam /= static_cast<double>(K_);
                H_k_tilde_gtsam /= static_cast<double>(K_);
            }
            constexpr double svn_hess_lambda = 1e-6;
            H_k_tilde_gtsam += svn_hess_lambda * I6;

             // Condition number calculation (Optional)
             // Note: std::cout inside a parallel loop can jumble output.
             // Consider removing or protecting this for non-debug runs.
             /*
             Eigen::JacobiSVD<Matrix6d> svd(H_k_tilde_gtsam);
             double cond = std::numeric_limits<double>::infinity();
             if (svd.singularValues()(svd.singularValues().size() - 1) > 1e-9) {
                 cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
             }
             #pragma omp critical
             {
                std::cout << "  Iter " << iter << ", k=" << k << ": H_tilde Cond. Num: " << cond << std::endl;
             }
             */

            // Solve linear system (remains the same)
            Eigen::LDLT<Matrix6d> solver(H_k_tilde_gtsam);
            if (solver.info() == Eigen::Success && H_k_tilde_gtsam.allFinite()) {
                 Vector6d update = solver.solve(-phi_k_star_gtsam);
                 if (update.allFinite()) { particle_updates[k] = update; }
                 else { /* Warn/Reset */ particle_updates[k].setZero(); }
            } else { /* Error/Reset */ particle_updates[k].setZero(); }
        } // End OpenMP Stage 2
        auto stage2_end_time = std::chrono::high_resolution_clock::now();

        // --- Stage 3: Apply Updates to Particles (Serial) ---
        // (Remains unchanged)
        auto stage3_start_time = std::chrono::high_resolution_clock::now();
        // --- MODIFICATION: Keep track of old update norm for logging ---
        double total_update_norm_sq = 0.0;
        // --- END MODIFICATION ---
        for (int k = 0; k < K_; ++k) {
            Vector6d scaled_update = step_size_ * particle_updates[k];
            if (!scaled_update.allFinite()) { continue; } // Skip NaN/Inf updates
            // --- MODIFICATION: Keep track of old update norm for logging ---
            total_update_norm_sq += particle_updates[k].squaredNorm();
            // --- END MODIFICATION ---
            particles[k] = particles[k].retract(scaled_update);
        }
        auto stage3_end_time = std::chrono::high_resolution_clock::now();
        // --- MODIFICATION: Keep track of old update norm for logging ---
        double avg_particle_update_norm = (K_ > 0) ? std::sqrt(total_update_norm_sq / static_cast<double>(K_)) : 0.0;
        // --- END MODIFICATION ---


        // --- MODIFICATION: Compute new mean pose ---
        // We use the simple Euclidean average in the tangent space at the *prior*
        // as it's a consistent reference point across iterations.
        Vector6d mean_xi_at_prior_current = Vector6d::Zero();
        for(int k=0; k < K_; ++k) {
            mean_xi_at_prior_current += gtsam::Pose3::Logmap(prior_mean.between(particles[k]));
        }
        if (K_ > 0) mean_xi_at_prior_current /= static_cast<double>(K_);
        mean_pose_current = prior_mean.retract(mean_xi_at_prior_current);
        // --- END MODIFICATION ---


        // --- MODIFICATION: Check Convergence & Timing ---
        result.iterations = iter + 1;
        
        // Calculate the norm of the update to the *mean pose* in its tangent space
        double mean_pose_update_norm = gtsam::Pose3::Logmap(mean_pose_last_iter.between(mean_pose_current)).norm();

        auto iter_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> stage1_ms = stage1_end_time - stage1_start_time;
        std::chrono::duration<double, std::milli> stage2_ms = stage2_end_time - stage2_start_time;
        std::chrono::duration<double, std::milli> stage3_ms = stage3_end_time - stage3_start_time;
        std::chrono::duration<double, std::milli> iter_ms = iter_end_time - iter_start_time;

        std::cout << std::fixed << std::setprecision(6);
        // Log both metrics
        std::cout << "[SVN Iter " << std::setw(2) << iter << "] Mean Pose Update: " << mean_pose_update_norm
                  << " | Avg Particle Update: " << avg_particle_update_norm
                  << " (T: " << std::setprecision(1) << iter_ms.count() << "ms = "
                  << "S1:" << stage1_ms.count() << " + S2:" << stage2_ms.count() << " + S3:" << stage3_ms.count() << ")" << std::endl;

        // Use the *new* norm for the check
        if (mean_pose_update_norm < stop_thresh_) {
            result.converged = true;
            std::cout << "[SvnNdt::align] Converged in " << result.iterations << " iterations (Mean Pose Update: " << mean_pose_update_norm << " < " << stop_thresh_ << ")." << std::endl;
            break;
        }
        // --- END MODIFICATION ---

    } // End SVN iteration loop

    // --- Finalization: Compute Mean Pose and Covariance ---
    // --- MODIFICATION: Use the already-computed mean pose ---
    result.final_pose = mean_pose_current;
    // --- END MODIFICATION ---

    if (K_ > 1) {
        result.final_covariance.setZero();
        // Recalculate tangent vectors at the *new final mean*
        std::vector<Vector6d> tangent_vectors_at_mean(K_);
         Vector6d mean_xi_at_mean = Vector6d::Zero();
         for(int k=0; k < K_; ++k) {
             tangent_vectors_at_mean[k] = gtsam::Pose3::Logmap(result.final_pose.between(particles[k]));
             mean_xi_at_mean += tangent_vectors_at_mean[k];
         }
         mean_xi_at_mean /= static_cast<double>(K_); // This should be near zero
        // Compute covariance
        for(int k=0; k < K_; ++k) {
            Vector6d diff = tangent_vectors_at_mean[k] - mean_xi_at_mean;
            result.final_covariance += diff * diff.transpose();
        }
        result.final_covariance /= static_cast<double>(K_ - 1);
    } else { // K_ == 1
         PCL_WARN("[SvnNdt::align] K=1, cannot compute sample covariance. Returning small diagonal covariance.\n");
         // Vector6d initial_sigmas; // Re-declare for scope (Already in original)
         initial_sigmas << 0.01, 0.01, 0.02, 0.05, 0.05, 0.05; 
         result.final_covariance = (1e-6 * initial_sigmas.array().square()).matrix().asDiagonal(); // Use square for variance
    }

    // Final Covariance Regularization (remains the same)
    Eigen::SelfAdjointEigenSolver<Matrix6d> final_eigensolver(result.final_covariance);
    if (final_eigensolver.info() == Eigen::Success) {
        Vector6d final_evals = final_eigensolver.eigenvalues();
        double min_eigenvalue = 1e-9;
        bool needs_regularization = false;
        for(int i=0; i<6; ++i) {
            if (final_evals(i) < min_eigenvalue) {
                final_evals(i) = min_eigenvalue;
                needs_regularization = true;
            }
        }
        if (needs_regularization) {
             result.final_covariance = final_eigensolver.eigenvectors() * final_evals.asDiagonal() * final_eigensolver.eigenvectors().transpose();
        }
    } else {
         PCL_WARN("[SVNNdt::align] Eigendecomposition failed for final covariance. Using regularized identity.\Vn");
         result.final_covariance = 1e-6 * I6;
    }


    input_.reset(); // Release shared pointer

    // Report if max iterations reached (remains the same)
    if (!result.converged && result.iterations >= max_iter_) {
        // --- MODIFICATION: Update log message ---
        std::cout << "[SvnNdt::align] Reached max iterations (" << max_iter_
                      << ") without converging (Last Mean Pose Update: " << std::fixed << std::setprecision(6) << gtsam::Pose3::Logmap(mean_pose_last_iter.between(mean_pose_current)).norm()
                      << " >= " << stop_thresh_ << ")." << std::endl;
        // --- END MODIFICATION ---
    }

    return result;
}

} // namespace svn_ndt

#endif // SVN_NDT_SVN_NDT_IMPL_HPP_