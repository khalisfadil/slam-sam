#ifndef SVN_NDT_SVN_NDT_IMPL_HPP_
#define SVN_NDT_SVN_NDT_IMPL_HPP_

// Include the class header
#include <svn_ndt.h> // Make sure this path is correct

// --- Standard/External Library Includes ---
#include <vector>
#include <cmath>
#include <numeric> // For std::accumulate (not used here but good practice)
#include <limits>
#include <iostream> // For PCL_WARN/ERROR and debug output
#include <chrono>   // For timing if needed
#include <iomanip> // For std::fixed, std::setprecision in debug output

// TBB for parallelism
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
// #include <tbb/spin_mutex.h> // Avoid if possible, design for lock-free

// Eigen for linear algebra
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Cholesky> // For LDLT solver

// PCL for point cloud operations
#include <pcl/common/transforms.h> // For pcl::transformPointCloud
#include <pcl/common/point_tests.h> // For pcl::isFinite
#include <pcl/io/pcd_io.h> // Include for PCL_WARN_STREAM

// GTSAM for pose representation and operations
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/numericalDerivative.h> // Not directly used, but good dependency check
// #include <gtsam/inference/Symbol.h> // Not needed for core logic
// #include <gtsam/nonlinear/Values.h> // Not needed for core logic
// #include <gtsam/nonlinear/NonlinearFactorGraph.h> // Not needed for core logic

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
      K_(30), max_iter_(50), kernel_h_(1.0), // Updated kernel_h default from config
      step_size_(0.0005), // Updated step_size default from config
      stop_thresh_(1e-4)
{
    updateNdtConstants(); // Initialize gauss_d* constants based on defaults
    // Zero out precomputed matrices initially
    j_ang_.setZero();
    h_ang_.setZero();
    // Debug print initial constants
    // std::cout << "Initial NDT Constants: d1=" << gauss_d1_ << ", d2=" << gauss_d2_ << ", d3=" << gauss_d3_ << std::endl;
}

template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::updateNdtConstants()
{
    // Recalculate NDT constants based on current resolution and outlier ratio
    // Based on NDT paper (Magnusson 2009) and common implementations (e.g., ndt_omp)
    if (resolution_ <= 1e-6f) { // Use epsilon for float comparison
        PCL_ERROR("[SvnNdt] Resolution must be positive. Cannot update NDT constants.\n");
        // Set safe defaults to prevent division by zero or log(0) later
        gauss_d1_ = 1.0; gauss_d2_ = 1.0; gauss_d3_ = 0.0;
        return;
    }

    // Intermediate constants based on outlier ratio and resolution
    double gauss_c1 = 10.0 * (1.0 - outlier_ratio_);
    double gauss_c2 = outlier_ratio_ / pow(static_cast<double>(resolution_), 3); // Use double pow

    // Add small epsilon to prevent log(0) or log(negative)
    constexpr double epsilon = 1e-9;
    if (gauss_c1 <= epsilon) gauss_c1 = epsilon;
    if (gauss_c2 <= epsilon) gauss_c2 = epsilon;

    double c1_plus_c2 = gauss_c1 + gauss_c2;
    // No need to check c1_plus_c2 <= epsilon as c1, c2 >= epsilon
    gauss_d3_ = -log(gauss_c2);
    gauss_d1_ = -log(c1_plus_c2) - gauss_d3_; // d1 = -log((c1+c2)/c2) = log(c2/(c1+c2))

    // Calculate d2 using term that should be exp(-d2/2) = (c1*exp(-0.5)+c2)/(c1+c2)
    double term_exp_neg_half = exp(-0.5); // exp(-1/2)
    double numerator_for_d2_log = gauss_c1 * term_exp_neg_half + gauss_c2;
    if (numerator_for_d2_log <= epsilon) numerator_for_d2_log = epsilon;

    double term_for_d2_log = numerator_for_d2_log / c1_plus_c2;
    // Check if the argument to log is valid
    if (term_for_d2_log <= epsilon) {
        PCL_WARN("[SvnNdt] Invalid argument for log in gauss_d2 calculation (ratio=%.3f). Using default d2=1.0.\n", term_for_d2_log);
        gauss_d2_ = 1.0; // Assign a default, perhaps 1.0 is safer than 2.0
    } else {
        // d2 = -2 * log( (c1*exp(-0.5)+c2)/(c1+c2) )
        // Note: The formula in the original code involving d1 and d3 seemed overly complex and prone to issues.
        // This direct calculation based on the NDT paper's score function definition is clearer.
        gauss_d2_ = -2.0 * log(term_for_d2_log);
    }

    // Final check for NaN or Inf (can happen with extreme inputs)
    if (!std::isfinite(gauss_d1_) || !std::isfinite(gauss_d2_) || !std::isfinite(gauss_d3_)) {
        PCL_ERROR("[SvnNdt] NaN/Inf detected in NDT constant calculation. Check resolution (%.3f) and outlier ratio (%.3f).\n", resolution_, outlier_ratio_);
        // Set safe defaults
        gauss_d1_ = 1.0; gauss_d2_ = 1.0; gauss_d3_ = 0.0;
    }
     // Optional Debug print
     // std::cout << std::fixed << std::setprecision(5);
     // std::cout << "Updated NDT Constants: d1=" << gauss_d1_ << ", d2=" << gauss_d2_ << ", d3=" << gauss_d3_
     //           << " (res=" << resolution_ << ", ratio=" << outlier_ratio_ << ")" << std::endl;
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
    // Set the input cloud for the VoxelGridCovariance instance
    target_cells_.setInputCloud(cloud);

    // If resolution is valid, configure and build the voxel grid
    if (resolution_ > 1e-6f) { // Use epsilon
        target_cells_.setLeafSize(resolution_, resolution_, resolution_);
        // Determine if the KdTree needs to be built based on the selected search method
        bool build_kdtree = (search_method_ == NeighborSearchMethod::KDTREE);
        // Build the grid (calculates means, covariances, etc.) and optionally the KdTree
        target_cells_.filter(build_kdtree);
        // Update NDT constants as they depend on the resolution used for the grid
        updateNdtConstants();
    } else {
        PCL_WARN("[SvnNdt::setInputTarget] Target cloud set, but resolution is not positive. Grid not built.\n");
    }
}

template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::setResolution(float resolution)
{
    if (resolution <= 1e-6f) { // Use epsilon
        PCL_ERROR("[SvnNdt::setResolution] Resolution must be positive.\n");
        return;
    }
    // Only update and rebuild if the resolution actually changes
    if (std::abs(resolution_ - resolution) > 1e-6f) {
        resolution_ = resolution;
        // If a target cloud exists, rebuild the grid with the new resolution
        if (target_cells_.getInputCloud()) {
            // This implicitly calls updateNdtConstants() inside setInputTarget
            setInputTarget(target_cells_.getInputCloud());
        }
    }
}

template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::setOutlierRatio(double ratio)
{
    double clamped_ratio = ratio;
    // Clamp ratio to the valid range [0, 1)
    if (ratio < 0.0) {
         PCL_WARN("[SvnNdt::setOutlierRatio] Outlier ratio must be non-negative. Clamping to 0.0.\n");
         clamped_ratio = 0.0;
    } else if (ratio >= 1.0) {
         PCL_WARN("[SvnNdt::setOutlierRatio] Outlier ratio must be less than 1.0. Clamping near 1.0.\n");
         clamped_ratio = 0.9999; // Clamp slightly below 1 to avoid potential issues
    }

    // Only update constants if the ratio actually changes
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
    // If bandwidth is effectively zero, kernel is delta function (1 if identical, 0 otherwise)
    // Here, return 1.0 for stability if poses are identical or h is tiny.
    if (kernel_h_ <= 1e-12) { // Use a smaller epsilon for kernel bandwidth check
        return (pose_l.equals(pose_k, 1e-9)) ? 1.0 : 0.0;
    }
    // Calculate difference vector in tangent space at pose_l
    Vector6d diff_log = gtsam::Pose3::Logmap(pose_l.between(pose_k)); // Logmap(T_l^{-1} * T_k)
    double sq_norm = diff_log.squaredNorm();
    return std::exp(-sq_norm / kernel_h_);
}

template <typename PointSource, typename PointTarget>
typename SvnNormalDistributionsTransform<PointSource, PointTarget>::Vector6d
SvnNormalDistributionsTransform<PointSource, PointTarget>::rbf_kernel_gradient(
    const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const
{
     // If bandwidth is effectively zero, gradient is zero
     if (kernel_h_ <= 1e-12) {
         return Vector6d::Zero();
     }
    // Calculate difference vector in tangent space at pose_l
    Vector6d diff_log = gtsam::Pose3::Logmap(pose_l.between(pose_k));
    double sq_norm = diff_log.squaredNorm();
    double k_val = std::exp(-sq_norm / kernel_h_);

    // Gradient of exp(-|log(Tl^-1 * Tk)|^2 / h) w.r.t. Tl (using tangent space approximation)
    // Gradient is k_val * (-2/h) * log(Tl^-1 * Tk)
    return k_val * (-2.0 / kernel_h_) * diff_log;
}


//=================================================================================================
// NDT Math Functions (Adapted SERIAL Implementations)
// These functions assume the NDT standard [x,y,z,roll,pitch,yaw] pose vector convention.
//=================================================================================================

// --- computeAngleDerivatives ---
// Precomputes angular components for Jacobian and Hessian based on [r,p,y] from the input p vector.
template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::computeAngleDerivatives(
    const Vector6d& p, bool compute_hessian) // p is expected [x,y,z,r,p,y]
{
    // Use roll, pitch, yaw from the input vector p
    double r = p(3), pi = p(4), y = p(5);

    // Simplified math for near 0 angles to improve numerical stability
    double cx, cy, cz, sx, sy, sz;
    constexpr double angle_epsilon = 1e-6; // Use a slightly larger epsilon for trig stability
    if (std::abs(r) < angle_epsilon) { sx = 0.0; cx = 1.0; } else { sx = sin(r); cx = cos(r); }
    if (std::abs(pi) < angle_epsilon) { sy = 0.0; cy = 1.0; } else { sy = sin(pi); cy = cos(pi); }
    if (std::abs(y) < angle_epsilon) { sz = 0.0; cz = 1.0; } else { sz = sin(y); cz = cos(y); }

    // --- Jacobian Components (Equation 6.19 [Magnusson 2009]) ---
    j_ang_.setZero(); // Ensure initialization
    // Derivatives w.r.t Roll (j_ang columns 0-2 for x,y,z components)
    j_ang_(0,0)=-sx*sz+cx*sy*cz; j_ang_(1,0)= cx*sz+sx*sy*cz; j_ang_(2,0)=-sy*cz;
    j_ang_(3,0)= sx*cy*cz;       j_ang_(4,0)=-cx*cy*cz;       j_ang_(5,0)=-cy*sz;
    j_ang_(6,0)= cx*cz-sx*sy*sz; j_ang_(7,0)= sx*cz+cx*sy*sz;
    // Derivatives w.r.t Pitch
    j_ang_(0,1)=-sx*cz-cx*sy*sz; j_ang_(1,1)= cx*cz-sx*sy*sz; j_ang_(2,1)= sy*sz;
    j_ang_(3,1)=-sx*cy*sz;       j_ang_(4,1)= cx*cy*sz;       j_ang_(5,1)=-cy*cz;
    j_ang_(6,1)=-cx*sz-sx*sy*cz; j_ang_(7,1)= cx*sy*cz-sx*sz;
    // Derivatives w.r.t Yaw
    j_ang_(0,2)=-cx*cy;          j_ang_(1,2)=-sx*cy;          j_ang_(2,2)= cy;
    j_ang_(3,2)= sx*sy;          j_ang_(4,2)=-cx*sy;          j_ang_(5,2)= 0.0;
    j_ang_(6,2)= 0.0;            j_ang_(7,2)= 0.0;


    // --- Hessian Components (Equation 6.21 [Magnusson 2009]) ---
    if (compute_hessian) {
        h_ang_.setZero(); // Ensure initialization
        // Calculate only the unique second derivative components
        h_ang_(0,0)=-cx*sz-sx*sy*cz; h_ang_(1,0)=-sx*sz+cx*sy*cz; // dR/drdr
        h_ang_(2,0)= cx*cy*cz;       h_ang_(3,0)= sx*cy*cz;       // dR/drdp
        h_ang_(4,0)=-sx*cz-cx*sy*sz; h_ang_(5,0)= cx*cz-sx*sy*sz; // dR/drdy

        h_ang_(6,1)=-cy*cz;          h_ang_(7,1)=-sx*sy*cz;       // dR/dpdp (Note: Swapped indices from original code for clarity)
        h_ang_(8,1)= cx*sy*cz;       // dR/dpdp

        h_ang_(9,1)= sy*sz;          h_ang_(10,1)=-sx*cy*sz;      // dR/dpdy
        h_ang_(11,1)= cx*cy*sz;

        h_ang_(12,2)=-cy*cz;         h_ang_(13,2)=-cx*sz-sx*sy*cz;// dR/dydy (Note: Swapped indices from original code for clarity)
        h_ang_(14,2)=-sx*sz+cx*sy*cz;
        // h_ang(15,?) seems unused

        // ** REMOVED FAULTY SYMMETRIC BLOCK ASSIGNMENTS **
        // Symmetry is handled in computePointDerivatives when filling point_hessian_
    }
}


// --- computePointDerivatives ---
// Computes Jacobian and Hessian of the transformed point coordinates w.r.t. the pose parameters p = [x,y,z,r,p,y]
template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::computePointDerivatives(
    const Eigen::Vector3d& x,                     // Original point coordinates
    Eigen::Matrix<float, 4, 6>& point_gradient_, // Output Jacobian Jp (4x6)
    Eigen::Matrix<float, 24, 6>& point_hessian_, // Output Hessian Hp (flattened 24x6)
    bool compute_hessian)
{
    // Point in homogeneous coordinates (float for consistency with matrix types)
    Eigen::Vector4f x4(static_cast<float>(x[0]), static_cast<float>(x[1]), static_cast<float>(x[2]), 0.0f); // w=0 for vector/point difference

    // --- Jacobian Calculation (Equation 6.18, 6.19 [Magnusson 2009]) ---
    // Jp = [ d(Tp)/dt | d(Tp)/da ] = [ I | d(Rx+t)/da ] = [ I | (dR/da)*x ]
    point_gradient_.setZero();
    point_gradient_.block<3, 3>(0, 0).setIdentity(); // Top-left 3x3 is Identity (derivative w.r.t translation)

    // Calculate angular part: (dR/da)*x by multiplying precomputed components by x
    Eigen::Matrix<float, 8, 1> x_j_ang = j_ang_ * x4; // (8x4) * (4x1) -> (8x1)

    // Map components of x_j_ang to the correct columns (3, 4, 5) of point_gradient_
    // Column 3: Derivatives w.r.t roll
    point_gradient_(0, 3) = x_j_ang[0]; point_gradient_(1, 3) = x_j_ang[1]; point_gradient_(2, 3) = x_j_ang[2];
    // Column 4: Derivatives w.r.t pitch
    point_gradient_(0, 4) = x_j_ang[3]; point_gradient_(1, 4) = x_j_ang[4]; point_gradient_(2, 4) = x_j_ang[5];
    // Column 5: Derivatives w.r.t yaw
    point_gradient_(0, 5) = x_j_ang[6]; point_gradient_(1, 5) = x_j_ang[7];
    // point_gradient_(2, 5) is implicitly zero based on j_ang_ calculation


    // --- Hessian Calculation (Equation 6.20, 6.21 [Magnusson 2009]) ---
    if (compute_hessian) {
        point_hessian_.setZero(); // Clear previous point's data
        // Calculate angular part: (d^2R/dadb)*x by multiplying precomputed components by x
        Eigen::Matrix<float, 16, 1> x_h_ang = h_ang_ * x4; // (16x4) * (4x1) -> (16x1)

        // Map components of x_h_ang to the correct blocks of point_hessian_
        // The point_hessian_ stores the 18x6 second derivative tensor flattened.
        // Each 4x1 block point_hessian_.block<4,1>(i*4, j) represents d^2(Tp)/dp_i dp_j * x
        // Indices: p = [t1,t2,t3, r1,r2,r3] -> parameter indices 0..5. NDT uses r1=idx 3, r2=idx 4, r3=idx 5
        // Re-verify mapping based on h_ang_ assignments and standard NDT Hessian structure

        // Hessian block H_rr (i=3, j=3): d^2(Tp)/dr dr * x uses h_ang rows 0, 1
        point_hessian_.block<3, 1>(3 * 4 + 1, 3) = x_h_ang.segment<2>(0); // y, z components from rows 0, 1
        // Hessian block H_rp (i=3, j=4): d^2(Tp)/dr dp * x uses h_ang rows 2, 3
        point_hessian_.block<3, 1>(3 * 4 + 1, 4) = x_h_ang.segment<2>(2); // y, z components from rows 2, 3
        // Hessian block H_ry (i=3, j=5): d^2(Tp)/dr dy * x uses h_ang rows 4, 5
        point_hessian_.block<3, 1>(3 * 4 + 1, 5) = x_h_ang.segment<2>(4); // y, z components from rows 4, 5

        // Hessian block H_pp (i=4, j=4): d^2(Tp)/dp dp * x uses h_ang rows 6, 7, 8
        point_hessian_.block<3, 1>(4 * 4 + 0, 4) = x_h_ang.segment<3>(6); // x, y, z components from rows 6, 7, 8
        // Hessian block H_py (i=4, j=5): d^2(Tp)/dp dy * x uses h_ang rows 9, 10, 11
        point_hessian_.block<3, 1>(4 * 4 + 0, 5) = x_h_ang.segment<3>(9); // x, y, z components from rows 9, 10, 11

        // Hessian block H_yy (i=5, j=5): d^2(Tp)/dy dy * x uses h_ang rows 12, 13, 14
        point_hessian_.block<3, 1>(5 * 4 + 0, 5) = x_h_ang.segment<3>(12); // x, y, z components from rows 12, 13, 14

        // Fill symmetric blocks (H_ij = H_ji)
        point_hessian_.block<4, 1>(4 * 4, 3) = point_hessian_.block<4, 1>(3 * 4, 4); // H_pr = H_rp
        point_hessian_.block<4, 1>(5 * 4, 3) = point_hessian_.block<4, 1>(3 * 4, 5); // H_yr = H_ry
        point_hessian_.block<4, 1>(5 * 4, 4) = point_hessian_.block<4, 1>(4 * 4, 5); // H_yp = H_py
    }
}


// --- updateDerivatives ---
// Calculates the contribution of a single point-voxel interaction to the NDT score, gradient, and Hessian.
template <typename PointSource, typename PointTarget>
double SvnNormalDistributionsTransform<PointSource, PointTarget>::updateDerivatives(
    Vector6d& score_gradient, // Accumulated Gradient (output)
    Matrix6d& hessian,        // Accumulated Hessian (output)
    const Eigen::Matrix<float, 4, 6>& point_gradient4, // Jacobian of point transform w.r.t p=[x,y,z,r,p,y]
    const Eigen::Matrix<float, 24, 6>& point_hessian_, // Hessian of point transform w.r.t p=[x,y,z,r,p,y]
    const Eigen::Vector3d& x_trans, // Point relative to voxel mean (point - mean)
    const Eigen::Matrix3d& c_inv,   // Voxel inverse covariance
    bool compute_hessian,
    bool print_debug) // Flag to enable detailed debug prints for this call
{
    // Use float types for intermediate calculations consistent with point_gradient/hessian types
    Eigen::Matrix<float, 1, 4> x_trans4(static_cast<float>(x_trans[0]), static_cast<float>(x_trans[1]), static_cast<float>(x_trans[2]), 0.0f);
    // Embed 3x3 inverse covariance into a 4x4 matrix
    Eigen::Matrix4f c_inv4 = Eigen::Matrix4f::Zero();
    c_inv4.topLeftCorner(3, 3) = c_inv.cast<float>();

    // Calculate Mahalanobis distance squared: (x-mu)^T * C^-1 * (x-mu)
    // Use double for potentially better precision in dot product
    double mahal_sq = x_trans.dot(c_inv * x_trans);

    // Check for invalid Mahalanobis distance or potential overflow in exponent
    // exp(-25) is already very small, avoid large positive exponents
    const double max_exponent_arg = 50.0;
    if (!std::isfinite(mahal_sq) || mahal_sq < -1e-9 || (gauss_d2_ * mahal_sq > max_exponent_arg)) { // Allow small negative due to float errors
        // if (print_debug) { // Keep this conditional for reducing noise
        //      std::cerr << "      updateDeriv WARN: mahal_sq invalid (" << mahal_sq << ") or exponent too large.\n";
        // }
        return 0.0; // Return zero score contribution for invalid points
    }
     // Clamp negative values slightly below zero before exponentiation if needed, although they shouldn't occur for PSD matrices
     if (mahal_sq < 0.0) mahal_sq = 0.0;


    // Calculate the exponent term: exp(-d2 * mahal^2 / 2)
    double exp_term = std::exp(-gauss_d2_ * mahal_sq * 0.5);

    // Calculate score increment: -d1 * exp_term (Equation 6.9 [Magnusson 2009])
    double score_inc = -gauss_d1_ * exp_term;

    // Calculate the common factor for gradient and Hessian: d1 * d2 * exp_term
    double factor = gauss_d1_ * gauss_d2_ * exp_term;

    // Check if factor is finite (could be NaN/Inf if exp_term was unstable)
    if (!std::isfinite(factor)) {
        // if (print_debug) { // Keep conditional
        //     std::cerr << "      updateDeriv WARN: factor is non-finite: " << factor << std::endl;
        // }
        return 0.0; // Return zero score contribution
    }

    // --- Gradient Calculation (Equation 6.12 [Magnusson 2009]) ---
    // grad_contrib = (x-mu)^T * C^-1 * Jp
    Eigen::Matrix<float, 4, 6> temp_vec = c_inv4 * point_gradient4; // C^-1 * Jp (4x6)
    Eigen::Matrix<float, 1, 6> grad_contrib_float = x_trans4 * temp_vec; // (x-mu)^T * C^-1 * Jp (1x6)

    // Increment overall gradient: factor * grad_contrib^T
    Vector6d grad_inc = factor * grad_contrib_float.transpose().cast<double>();

    // --- DETAILED DEBUG PRINT (BEFORE accumulating) ---
    if (print_debug) {
         std::cout << std::fixed << std::setprecision(5);
         std::cout << "      updateDeriv [DBG]: mahal^2=" << mahal_sq
                   << ", exp_t=" << exp_term
                   << ", d1=" << gauss_d1_ << ", d2=" << gauss_d2_
                   << ", factor=" << factor << std::endl;
         std::cout << "                     : c_inv.n=" << c_inv.norm()
                   << ", pt_grad.n=" << point_gradient4.norm()
                   << ", grad_contr.n=" << grad_contrib_float.norm()
                   << ", grad_inc.n=" << grad_inc.norm() << std::endl;
    }
    // --- END DEBUG PRINT ---

    // Check gradient increment validity before adding
    if (!grad_inc.allFinite()){
        // if (print_debug) { // Keep conditional
        //     std::cerr << "      updateDeriv WARN: grad_inc is non-finite!" << std::endl;
        // }
        return 0.0; // Don't add invalid gradient and return zero score
    }
    score_gradient += grad_inc;


    // --- Hessian Calculation (Equation 6.13 [Magnusson 2009], using approximation) ---
    if (compute_hessian) {
        Matrix6d hess_contrib = Matrix6d::Zero(); // Accumulate contribution for this point-voxel pair
        Eigen::Matrix<double, 1, 6> grad_contrib_double = grad_contrib_float.cast<double>();

        // Term 1: -d2 * [(x-mu)^T * C^-1 * Jp]^T * [(x-mu)^T * C^-1 * Jp]
        hess_contrib = -gauss_d2_ * (grad_contrib_double.transpose() * grad_contrib_double);

        // Term 2: Jp^T * C^-1 * Jp
        hess_contrib += (point_gradient4.transpose() * temp_vec).cast<double>(); // temp_vec = C^-1 * Jp

        // Term 3: (x-mu)^T * C^-1 * Hp
        Eigen::Matrix<float, 1, 4> x_trans4_c_inv4 = x_trans4 * c_inv4; // Precompute (x-mu)^T * C^-1
        Matrix6d term3 = Matrix6d::Zero();
        for (int i = 0; i < 6; ++i) { // Row index of Hessian
            for (int j = i; j < 6; ++j) { // Column index of Hessian (use symmetry)
                 // Extract the (i,j) block (4x1 vector) from the flattened point Hessian Hp
                 // This block represents d^2(Tp)/dp_i dp_j * x
                 // ** Corrected block access based on 4 rows per parameter **
                 Eigen::Matrix<float, 4, 1> H_ij_x = point_hessian_.block<4, 1>(i * 4, j);
                 // Calculate the scalar contribution (x-mu)^T * C^-1 * [d^2(Tp)/dp_i dp_j * x]
                 term3(i, j) = x_trans4_c_inv4 * H_ij_x;
            }
        }
        // Fill the lower triangle using symmetry
        term3.template triangularView<Eigen::Lower>() = term3.template triangularView<Eigen::Upper>().transpose();
        hess_contrib += term3;

        // Scale total contribution by factor
        hess_contrib *= factor;

        // Check Hessian contribution validity before adding
        if (!hess_contrib.allFinite()){
            // if (print_debug) { // Keep conditional
            //     std::cerr << "      updateDeriv WARN: hess_contrib is non-finite!" << std::endl;
            // }
            // If hessian is invalid, maybe just skip adding it? Or add only Term 2 scaled?
            // Skipping seems safer if the source of NaN/Inf is Term 1 or 3.
        } else {
            hessian += hess_contrib;
        }

        // --- Optional Debug Print for Hessian ---
        // if (print_debug) {
        //     std::cout << "                     : hess_contr.n=" << hess_contrib.norm() << std::endl;
        // }
        // --- End Optional Debug Print ---
    }


    return score_inc; // Return the score contribution of this point-voxel interaction
}


// --- computeParticleDerivatives ---
// Computes the total NDT score, gradient, and Hessian for a given pose p=[x,y,z,r,p,y]
template <typename PointSource, typename PointTarget>
double SvnNormalDistributionsTransform<PointSource, PointTarget>::computeParticleDerivatives(
    Vector6d& score_gradient, // Output gradient
    Matrix6d& hessian,        // Output Hessian
    const PointCloudSource& trans_cloud, // Transformed source cloud
    const Vector6d& p,        // Current pose estimate [x,y,z,r,p,y]
    bool compute_hessian)
{
    // --- Initializations ---
    score_gradient.setZero();
    hessian.setZero();
    double total_score = 0.0;
    bool first_point_processed_debug = false; // Flag for debug prints in updateDerivatives

    // --- Precompute Angle Derivatives ---
    // This depends on p=[x,y,z,r,p,y] and stores results in j_ang_ and h_ang_
    computeAngleDerivatives(p, compute_hessian);

    // --- Reusable Structures for inner loop ---
    Eigen::Matrix<float, 4, 6> point_gradient4; // Jacobian of point transform
    Eigen::Matrix<float, 24, 6> point_hessian24; // Hessian of point transform
    std::vector<LeafConstPtr> neighborhood; // Voxel neighbors for a point
    std::vector<float> distances; // Distances to neighbors (used by KDTREE search)
    constexpr size_t reserve_size = 27; // Max neighbors for DIRECT26 (more than DIRECT7)
    neighborhood.reserve(reserve_size);
    distances.reserve(reserve_size);

    // --- Loop Over Transformed Source Points ---
    for (size_t idx = 0; idx < trans_cloud.points.size(); ++idx)
    {
        const PointSource& x_trans_pt = trans_cloud.points[idx]; // Point already transformed by pose p
        // Skip invalid points
        if (!pcl::isFinite(x_trans_pt)) continue;

        // --- Neighbor Search ---
        // Find valid NDT voxels (LeafConstPtr) near the transformed point
        neighborhood.clear(); // Clear from previous point
        distances.clear();    // Clear from previous point
        int neighbors_found = 0;
        switch (search_method_)
        {
            case NeighborSearchMethod::KDTREE:
                // Use KdTree on voxel centroids (requires VoxelGridCovariance::filter(true))
                neighbors_found = target_cells_.radiusSearch(x_trans_pt, resolution_, neighborhood, distances);
                break;
            case NeighborSearchMethod::DIRECT7:
                // Check center + 6 face neighbors
                neighbors_found = target_cells_.getNeighborhoodAtPoint7(x_trans_pt, neighborhood);
                break;
            case NeighborSearchMethod::DIRECT1:
            default:
                // Check only the voxel containing the point
                neighbors_found = target_cells_.getNeighborhoodAtPoint1(x_trans_pt, neighborhood);
                break;
        }
        // Skip point if no valid neighbors (voxels with enough points) are found
        if (neighbors_found == 0) continue;

        // --- Get Original Point Coordinates ---
        // Need original coords for calculating derivatives w.r.t pose parameters
        if (!input_ || idx >= input_->size()) {
             PCL_ERROR("[SvnNdt::computeParticleDerivatives] Internal error: input_ cloud invalid or index out of bounds (%zu).\n", idx);
             continue; // Skip this point if source cloud pointer is bad
        }
        const PointSource& x_pt = (*input_)[idx]; // Original point
        Eigen::Vector3d x_orig(x_pt.x, x_pt.y, x_pt.z);

        // --- Compute Point Derivatives (Jacobian/Hessian w.r.t. pose p) ---
        // Uses x_orig and the precomputed j_ang_, h_ang_ (based on p)
        computePointDerivatives(x_orig, point_gradient4, point_hessian24, compute_hessian);

        // --- Accumulate Contributions from Neighbors ---
        double point_score_contribution = 0.0;
        Vector6d point_gradient_contribution = Vector6d::Zero();
        Matrix6d point_hessian_contribution = Matrix6d::Zero();
        bool is_first_point_for_debug = !first_point_processed_debug; // Flag for printing details only once

        for (const LeafConstPtr& cell : neighborhood)
        {
            if (!cell) { // Should not happen if neighbor search returns valid pointers
                 PCL_WARN("[SvnNdt::computeParticleDerivatives] Nullptr encountered in neighborhood.\n");
                 continue;
            }

            // Calculate point relative to voxel mean: x_trans = point_transformed - voxel_mean
            Eigen::Vector3d x_rel = Eigen::Vector3d(x_trans_pt.x, x_trans_pt.y, x_trans_pt.z) - cell->getMean();
            // Get precomputed inverse covariance from the voxel leaf
            const Eigen::Matrix3d& c_inv = cell->getInverseCov();

            // --- Call UpdateDerivatives to get contribution of this point-voxel interaction ---
            double score_inc = updateDerivatives(point_gradient_contribution, point_hessian_contribution,
                                                 point_gradient4, point_hessian24,
                                                 x_rel, c_inv, compute_hessian,
                                                 is_first_point_for_debug && (cell == neighborhood[0]) ); // Print debug only for the first point & first neighbor
            point_score_contribution += score_inc;

        } // End loop over neighbors

        // --- Update Debug Flag ---
        if (is_first_point_for_debug) {
             first_point_processed_debug = true;
        }

        // --- Add Point's Total Contribution to Overall Gradient/Hessian ---
        // Check for NaN/Inf before accumulating to prevent corrupting totals
        if (std::isfinite(point_score_contribution) &&
            point_gradient_contribution.allFinite() &&
            (!compute_hessian || point_hessian_contribution.allFinite()))
        {
            total_score += point_score_contribution;
            score_gradient += point_gradient_contribution;
            if (compute_hessian) {
                hessian += point_hessian_contribution;
            }
        } else {
             // Reduce verbosity, maybe print only once or count occurrences
             // PCL_WARN("[SvnNdt::computeParticleDerivatives] NaN/Inf detected in contribution for point index %zu. Skipping contribution.\n", idx);
        }

    } // End loop over points

    // --- Hessian Regularization (Levenberg-Marquardt style) ---
    // Add a small identity matrix scaled by lambda to the diagonal to improve conditioning
    if (compute_hessian) {
        constexpr double lambda = 1e-3; // Regularization strength (can be tuned)
        hessian += lambda * Matrix6d::Identity();
    }

    // --- Final Sanity Checks & Debug Print ---
    if (!score_gradient.allFinite()) {
        PCL_ERROR("[SvnNdt::computeParticleDerivatives] Final score_gradient contains NaN/Inf! Resetting to zero.\n");
        score_gradient.setZero(); // Avoid propagating errors
    }
    if (compute_hessian && !hessian.allFinite()) {
        PCL_ERROR("[SvnNdt::computeParticleDerivatives] Final hessian contains NaN/Inf! Resetting to identity.\n");
        hessian = Matrix6d::Identity(); // Use identity if invalid
    }

    // --- Debug Print for Final Gradient Norm of this particle ---
    // std::cout << std::fixed << std::setprecision(5);
    // std::cout << "    computeParticleDeriv Final Grad Norm: " << score_gradient.norm() << std::endl;
    // ---

    // Return the total NDT score (sum of -d1*exp(...) terms)
    return total_score;

} // End computeParticleDerivatives

//=================================================================================================
// Main Alignment Function (SVN-NDT Implementation)
//=================================================================================================
template <typename PointSource, typename PointTarget>
SvnNdtResult SvnNormalDistributionsTransform<PointSource, PointTarget>::align(
    const PointCloudSource& source_cloud,
    const gtsam::Pose3& prior_mean)
{
    SvnNdtResult result; // Structure to store alignment results

    // --- Input Sanity Checks ---
    if (!target_cells_.getInputCloud() || target_cells_.getAllLeaves().empty()) {
        PCL_ERROR("[SvnNdt::align] Target NDT grid is not initialized. Call setInputTarget() first.\n");
        result.converged = false;
        return result;
    }
    if (source_cloud.empty()) {
        PCL_ERROR("[SvnNdt::align] Input source cloud is empty.\n");
        result.converged = false;
        return result;
    }
    if (K_ <= 0) {
        PCL_ERROR("[SvnNdt::align] Particle count (K_) must be positive.\n");
        result.converged = false; // K_ is already clamped in setter, but double-check
        return result;
    }

    // --- Initialization ---
    // Store a const shared pointer to the source cloud for access within derivative calculations
    input_ = source_cloud.makeShared();

    // Initialize particle poses around the prior mean
    std::vector<gtsam::Pose3> particles(K_);
    // Define initial noise sigmas for particle spread (tune these if needed)
    // Order: [roll, pitch, yaw, x, y, z] in radians and meters
    Vector6d initial_sigmas; initial_sigmas << 0.02, 0.02, 0.05, 0.1, 0.1, 0.1; // Example values
    auto prior_noise_model = gtsam::noiseModel::Diagonal::Sigmas(initial_sigmas);
    gtsam::Sampler sampler(prior_noise_model, std::chrono::system_clock::now().time_since_epoch().count()); // Use time-based seed

    for (int k = 0; k < K_; ++k) {
        // Sample in the tangent space at prior_mean and retract to get a pose on the manifold
        particles[k] = prior_mean.retract(sampler.sample());
    }

    // Allocate storage for intermediate results (use aligned allocators for Eigen types)
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> loss_gradients(K_); // Gradient of NDT score * (-1)
    std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>> loss_hessians(K_);  // Hessian of NDT score * (-1)
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> particle_updates(K_); // Solved SVN update vectors
    std::vector<PointCloudSource> transformed_clouds(K_); // Per-particle transformed clouds

    Matrix6d I6 = Matrix6d::Identity(); // Reusable identity matrix

    // --- SVN Iteration Loop ---
    // ** FIX 2: Declare avg_update_norm outside the loop **
    double avg_update_norm = std::numeric_limits<double>::max(); // Initialize with large value
    for (int iter = 0; iter < max_iter_; ++iter)
    {
        auto iter_start_time = std::chrono::high_resolution_clock::now(); // For timing

        // --- Stage 1: Compute NDT Derivatives for each particle (Parallel using TBB) ---
        auto stage1_start_time = std::chrono::high_resolution_clock::now();
        tbb::parallel_for(tbb::blocked_range<int>(0, K_),
            [&](const tbb::blocked_range<int>& r) {

            // Create a thread-local copy of 'this' to manage internal state like j_ang_, h_ang_ safely.
            SvnNormalDistributionsTransform<PointSource, PointTarget> local_ndt = *this;
            local_ndt.input_ = this->input_; // Ensure the local copy points to the same input cloud

            Vector6d grad_k; // Per-particle gradient
            Matrix6d hess_k; // Per-particle Hessian

            for (int k = r.begin(); k < r.end(); ++k) {
                // Transform the original source cloud by the current particle's pose
                pcl::transformPointCloud(source_cloud, transformed_clouds[k], particles[k].matrix().cast<float>());

                // Convert gtsam::Pose3 particle to NDT's expected [x,y,z,r,p,y] vector
                Vector6d p_k_ndt;
                gtsam::Vector3 rpy = particles[k].rotation().rpy(); // GTSAM default RPY
                // ** FIX 1: Correctly get translation vector **
                p_k_ndt.head<3>() = particles[k].translation(); // gtsam::Point3 is compatible
                p_k_ndt.tail<3>() = rpy; // Assign roll, pitch, yaw

                // Compute NDT score, gradient, and Hessian for this particle's pose
                double score_k = local_ndt.computeParticleDerivatives(grad_k, hess_k, transformed_clouds[k], p_k_ndt, true);

                // Store NEGATED gradient and Hessian
                loss_gradients[k] = -grad_k;
                loss_hessians[k] = -hess_k;

                 // Sanity check the Hessian computed by NDT before storing
                 if (!hess_k.allFinite() || hess_k.hasNaN()) {
                     PCL_WARN("[SvnNdt::align Stage1] NaN/Inf in NDT Hessian for particle %d, iter %d. Using -Identity.\n", k, iter);
                     loss_hessians[k] = -I6; // Store negative identity if invalid
                 } else if (loss_hessians[k].hasNaN()) {
                     // This check might be redundant now, but keep for safety
                     PCL_WARN("[SvnNdt::align Stage1] Stored loss_hessian has NaN for particle %d, iter %d after negation. Using -Identity.\n", k, iter);
                     loss_hessians[k] = -I6;
                 }
            }
        }); // End TBB Stage 1
        auto stage1_end_time = std::chrono::high_resolution_clock::now();

        // --- Stage 2: Calculate SVN Updates (Parallel using TBB) ---
        auto stage2_start_time = std::chrono::high_resolution_clock::now();
        tbb::parallel_for(tbb::blocked_range<int>(0, K_),
            [&](const tbb::blocked_range<int>& r) {

            for (int k = r.begin(); k < r.end(); ++k) {
                Vector6d phi_k_star = Vector6d::Zero(); // SVGD direction component
                Matrix6d H_k_tilde = Matrix6d::Zero();  // SVN Hessian component

                // Aggregate contributions from all particles (l) to particle k
                for (int l = 0; l < K_; ++l) {
                    // Kernel value and gradient (operate on gtsam::Pose3)
                    double k_val = rbf_kernel(particles[l], particles[k]);
                    Vector6d k_grad = rbf_kernel_gradient(particles[l], particles[k]); // Gradient w.r.t particles[l]

                    // Check for numerical issues in kernel calculations
                    if (!std::isfinite(k_val) || !k_grad.allFinite()) {
                         PCL_WARN("[SvnNdt::align Stage2] NaN/Inf in kernel computation between particles %d and %d, iter %d.\n", l, k, iter);
                         continue; // Skip contribution from this pair
                     }

                    // Accumulate SVGD direction term: k(l,k)*grad(loss_l) + grad_l(k(l,k))
                    if (!loss_gradients[l].allFinite()) {
                         PCL_WARN("[SvnNdt::align Stage2] NaN/Inf in loss_gradient for particle %d, iter %d. Skipping term in phi_k_star.\n", l, iter);
                    } else {
                         phi_k_star += k_val * loss_gradients[l];
                    }
                    phi_k_star += k_grad; // Always add kernel gradient


                    // Accumulate SVN Hessian term: k(l,k)^2 * hess(loss_l) + grad_l(k(l,k)) * grad_l(k(l,k))^T
                    if (loss_hessians[l].allFinite()) { // Use the stored (negated) NDT hessian
                        H_k_tilde += (k_val * k_val) * loss_hessians[l];
                    } else {
                        // Optionally add a warning if Hessian was invalid
                        // PCL_WARN("[SvnNdt::align Stage2] Skipping invalid loss_hessian contribution from l=%d for H_k_tilde.\n", l);
                    }
                    // Always add the kernel gradient term
                    H_k_tilde += (k_grad * k_grad.transpose());
                }

                // Average over particles
                if (K_ > 0) {
                    phi_k_star /= static_cast<double>(K_);
                    H_k_tilde /= static_cast<double>(K_);
                }

                // Add regularization to the SVN Hessian for stability before inversion
                constexpr double svn_hess_lambda = 1e-4;
                H_k_tilde += svn_hess_lambda * I6;

                // Solve the linear system: H_k_tilde * update = phi_k_star
                Eigen::LDLT<Matrix6d> solver(H_k_tilde);
                if (solver.info() == Eigen::Success && H_k_tilde.allFinite()) {
                     Vector6d update = solver.solve(phi_k_star);
                     if (update.allFinite()) {
                         particle_updates[k] = update; // Store the solved update direction
                     } else {
                         PCL_WARN("[SvnNdt::align Stage2] Solver produced NaN/Inf update for particle %d, iter %d. Setting update to zero.\n", k, iter);
                         particle_updates[k].setZero();
                     }
                } else {
                    PCL_ERROR("[SvnNdt::align Stage2] LDLT solver failed or H_tilde invalid for particle %d, iter %d (Info: %d). Setting update to zero.\n", k, iter, solver.info());
                    // Optionally print H_k_tilde here for debugging
                    // if (!H_k_tilde.allFinite()) std::cerr << "H_k_tilde contained NaN/Inf!" << std::endl;
                    // else std::cerr << "H_k_tilde:\n" << H_k_tilde << std::endl;
                    particle_updates[k].setZero();
                }
            }
        }); // End TBB Stage 2
        auto stage2_end_time = std::chrono::high_resolution_clock::now();

        // --- Stage 3: Apply Updates to Particles (Serial) ---
        double total_update_norm_sq = 0.0; // Use squared norm initially to avoid sqrt
        for (int k = 0; k < K_; ++k) {
            // Apply the update direction scaled by step size
            // Following SVN paper Eq 16: xi <- xi + eps * H^-1 * phi*
            Vector6d scaled_update = step_size_ * particle_updates[k]; // Use positive update

             // Check for numerical issues in the final update vector
             if (!scaled_update.allFinite()) {
                 PCL_WARN("[SvnNdt::align Stage3] NaN/Inf in scaled update for particle %d, iter %d. Skipping update.\n", k, iter);
                 continue; // Skip updating this particle
             }

            // Accumulate squared norm of the *unscaled* update for convergence check
            total_update_norm_sq += particle_updates[k].squaredNorm();

            // Apply the update on the manifold using gtsam::Pose3::retract
            particles[k] = particles[k].retract(scaled_update);
        }
        auto stage3_end_time = std::chrono::high_resolution_clock::now();

        // --- Check Convergence ---
        result.iterations = iter + 1;
        // Calculate the average norm here
        avg_update_norm = (K_ > 0) ? std::sqrt(total_update_norm_sq / static_cast<double>(K_)) : 0.0;

        auto iter_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> stage1_ms = stage1_end_time - stage1_start_time;
        std::chrono::duration<double, std::milli> stage2_ms = stage2_end_time - stage2_start_time;
        std::chrono::duration<double, std::milli> stage3_ms = stage3_end_time - stage2_end_time; // Stage 3 starts after Stage 2 ends
        std::chrono::duration<double, std::milli> iter_ms = iter_end_time - iter_start_time;

        // Debug print for average update norm and timings
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "[SVN Iter " << std::setw(2) << iter << "] Avg Update Norm: " << avg_update_norm
                  << " (T: " << std::setprecision(1) << iter_ms.count() << "ms = "
                  << "S1:" << stage1_ms.count() << " + S2:" << stage2_ms.count() << " + S3:" << stage3_ms.count() << ")" << std::endl;

        // Check if average update norm is below the threshold
        if (avg_update_norm < stop_thresh_) {
            result.converged = true;
             std::cout << "[SvnNdt::align] Converged in " << result.iterations << " iterations (Avg Update Norm: " << avg_update_norm << " < " << stop_thresh_ << ")." << std::endl;
            break; // Exit loop
        }

    } // End SVN iteration loop

    // --- Finalization: Compute Mean Pose and Covariance from Particles ---
    if (K_ > 0) {
        // --- Calculate Mean Pose ---
        // Use Frechet mean approximation: average in tangent space of initial prior, then retract.
        Vector6d mean_xi_at_prior = Vector6d::Zero();
        for(int k=0; k < K_; ++k) {
            // Calculate tangent vector from prior_mean to particle k
            mean_xi_at_prior += gtsam::Pose3::Logmap(prior_mean.between(particles[k]));
        }
        mean_xi_at_prior /= static_cast<double>(K_);
        result.final_pose = prior_mean.retract(mean_xi_at_prior);

        // --- Calculate Covariance ---
        if (K_ > 1) {
            result.final_covariance.setZero();
            // Calculate tangent vectors relative to the *computed mean pose*
            std::vector<Vector6d> tangent_vectors_at_mean(K_);
             Vector6d mean_xi_at_mean = Vector6d::Zero(); // Should be near zero if mean is correct
             for(int k=0; k < K_; ++k) {
                 tangent_vectors_at_mean[k] = gtsam::Pose3::Logmap(result.final_pose.between(particles[k]));
                 mean_xi_at_mean += tangent_vectors_at_mean[k];
             }
             mean_xi_at_mean /= static_cast<double>(K_); // Calculate mean in the final tangent space

            // Compute sample covariance in the tangent space at the mean pose
            for(int k=0; k < K_; ++k) {
                Vector6d diff = tangent_vectors_at_mean[k] - mean_xi_at_mean; // Difference from mean tangent vector
                result.final_covariance += diff * diff.transpose();
            }
            result.final_covariance /= static_cast<double>(K_ - 1); // Use N-1 for sample covariance
        } else {
            // If only one particle, covariance is undefined from samples, use prior's covariance
            result.final_covariance = prior_noise_model->covariance(); // Or set to Identity/Zero? Prior seems reasonable.
        }

        // --- Final Covariance Regularization ---
        // Ensure the final covariance is positive semi-definite and reasonably conditioned
        Eigen::SelfAdjointEigenSolver<Matrix6d> final_eigensolver(result.final_covariance);
        if (final_eigensolver.info() == Eigen::Success) {
            Vector6d final_evals = final_eigensolver.eigenvalues();
            // Check if smallest eigenvalue is too small or negative
            if (final_evals(0) < 1e-9) { // Use a small threshold
                 PCL_DEBUG("[SvnNdt::align] Final covariance has small/negative eigenvalues. Applying regularization.\n");
                 // Inflate eigenvalues below threshold
                 for(int i=0; i<6; ++i) final_evals(i) = std::max(final_evals(i), 1e-9);
                 // Recompose
                 result.final_covariance = final_eigensolver.eigenvectors() * final_evals.asDiagonal() * final_eigensolver.eigenvectors().transpose();
            }
        } else {
             PCL_WARN("[SvnNdt::align] Eigendecomposition failed for final covariance. Matrix might be invalid. Using regularized identity.\n");
             result.final_covariance = 1e-6 * I6; // Use a small regularized identity as fallback
        }

    } else { // Should not happen due to check at start, but handle defensively
        result.final_pose = prior_mean;
        result.final_covariance.setIdentity(); // Default to identity if K=0
    }

    // Clear the internal pointer to the source cloud
    input_.reset();

    // ** FIX 2: Correct access to avg_update_norm and use PCL_WARN_STREAM **
    if (!result.converged && result.iterations >= max_iter_) {
        // Use PCL_WARN_STREAM for safer, stream-based output
        PCL_WARN_STREAM("[SvnNdt::align] Reached max iterations (" << max_iter_
                      << ") without converging (Avg Update Norm: " << std::fixed << std::setprecision(6) << avg_update_norm
                      << " >= " << stop_thresh_ << ").\n");
    }

    return result;
}

} // namespace svn_ndt

#endif // SVN_NDT_SVN_NDT_IMPL_HPP_