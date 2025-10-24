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
      K_(30), max_iter_(50), kernel_h_(1.0), 
      step_size_(0.0005), // You may need to INCREASE this after the fix
      stop_thresh_(1e-4)
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
    // Based on NDT paper (Magnusson 2009)
    if (resolution_ <= 1e-6f) { // Use epsilon for float comparison
        PCL_ERROR("[SvnNdt] Resolution must be positive. Cannot update NDT constants.\n");
        gauss_d1_ = 1.0; gauss_d2_ = 1.0; gauss_d3_ = 0.0;
        return;
    }

    double gauss_c1 = 10.0 * (1.0 - outlier_ratio_);
    double gauss_c2 = outlier_ratio_ / pow(static_cast<double>(resolution_), 3); // Use double pow

    constexpr double epsilon = 1e-9;
    if (gauss_c1 <= epsilon) gauss_c1 = epsilon;
    if (gauss_c2 <= epsilon) gauss_c2 = epsilon;

    double c1_plus_c2 = gauss_c1 + gauss_c2;
    gauss_d3_ = -log(gauss_c2);
    gauss_d1_ = -log(c1_plus_c2) - gauss_d3_; 

    double term_exp_neg_half = exp(-0.5); // exp(-1/2)
    double numerator_for_d2_log = gauss_c1 * term_exp_neg_half + gauss_c2;
    if (numerator_for_d2_log <= epsilon) numerator_for_d2_log = epsilon;

    double term_for_d2_log = numerator_for_d2_log / c1_plus_c2;
    if (term_for_d2_log <= epsilon) {
        PCL_WARN("[SvnNdt] Invalid argument for log in gauss_d2 calculation (ratio=%.3f). Using default d2=1.0.\n", term_for_d2_log);
        gauss_d2_ = 1.0; 
    } else {
        gauss_d2_ = -2.0 * log(term_for_d2_log);
    }

    if (!std::isfinite(gauss_d1_) || !std::isfinite(gauss_d2_) || !std::isfinite(gauss_d3_)) {
        PCL_ERROR("[SvnNdt] NaN/Inf detected in NDT constant calculation. Check resolution (%.3f) and outlier ratio (%.3f).\n", resolution_, outlier_ratio_);
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
        target_cells_.filter(build_kdtree);
        updateNdtConstants();
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
    if (std::abs(resolution_ - resolution) > 1e-6f) {
        resolution_ = resolution;
        if (target_cells_.getInputCloud()) {
            setInputTarget(target_cells_.getInputCloud());
        }
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
         PCL_WARN("[SvnNdt::setOutlierRatio] Outlier ratio must be less than 1.0. Clamping near 1.0.\n");
         clamped_ratio = 0.9999; 
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
    // These functions are already correct, as they operate purely in GTSAM space
    if (kernel_h_ <= 1e-12) { 
        return (pose_l.equals(pose_k, 1e-9)) ? 1.0 : 0.0;
    }
    // Logmap(T_l^{-1} * T_k) -> tangent vector at T_l
    Vector6d diff_log = gtsam::Pose3::Logmap(pose_l.between(pose_k)); 
    double sq_norm = diff_log.squaredNorm();
    return std::exp(-sq_norm / kernel_h_);
}

template <typename PointSource, typename PointTarget>
typename SvnNormalDistributionsTransform<PointSource, PointTarget>::Vector6d
SvnNormalDistributionsTransform<PointSource, PointTarget>::rbf_kernel_gradient(
    const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const
{
     // These functions are already correct, as they operate purely in GTSAM space
     if (kernel_h_ <= 1e-12) {
         return Vector6d::Zero();
     }
    Vector6d diff_log = gtsam::Pose3::Logmap(pose_l.between(pose_k));
    double sq_norm = diff_log.squaredNorm();
    double k_val = std::exp(-sq_norm / kernel_h_);

    // Gradient w.r.t. pose_l
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
    double r = p(3), pi = p(4), y = p(5);

    double cx, cy, cz, sx, sy, sz;
    constexpr double angle_epsilon = 1e-6; 
    if (std::abs(r) < angle_epsilon) { sx = 0.0; cx = 1.0; } else { sx = sin(r); cx = cos(r); }
    if (std::abs(pi) < angle_epsilon) { sy = 0.0; cy = 1.0; } else { sy = sin(pi); cy = cos(pi); }
    if (std::abs(y) < angle_epsilon) { sz = 0.0; cz = 1.0; } else { sz = sin(y); cz = cos(y); }

    // --- Jacobian Components ---
    j_ang_.setZero(); 
    j_ang_(0,0)=-sx*sz+cx*sy*cz; j_ang_(1,0)= cx*sz+sx*sy*cz; j_ang_(2,0)=-sy*cz;
    j_ang_(3,0)= sx*cy*cz;       j_ang_(4,0)=-cx*cy*cz;       j_ang_(5,0)=-cy*sz;
    j_ang_(6,0)= cx*cz-sx*sy*sz; j_ang_(7,0)= sx*cz+cx*sy*sz;
    j_ang_(0,1)=-sx*cz-cx*sy*sz; j_ang_(1,1)= cx*cz-sx*sy*sz; j_ang_(2,1)= sy*sz;
    j_ang_(3,1)=-sx*cy*sz;       j_ang_(4,1)= cx*cy*sz;       j_ang_(5,1)=-cy*cz;
    j_ang_(6,1)=-cx*sz-sx*sy*cz; j_ang_(7,1)= cx*sy*cz-sx*sz;
    j_ang_(0,2)=-cx*cy;          j_ang_(1,2)=-sx*cy;          j_ang_(2,2)= cy;
    j_ang_(3,2)= sx*sy;          j_ang_(4,2)=-cx*sy;          j_ang_(5,2)= 0.0;
    j_ang_(6,2)= 0.0;            j_ang_(7,2)= 0.0;


    // --- Hessian Components ---
    if (compute_hessian) {
        h_ang_.setZero(); 
        h_ang_(0,0)=-cx*sz-sx*sy*cz; h_ang_(1,0)=-sx*sz+cx*sy*cz; // dR/drdr (y,z components)
        h_ang_(2,0)= cx*cy*cz;       h_ang_(3,0)= sx*cy*cz;       // dR/drdp (y,z components)
        h_ang_(4,0)=-sx*cz-cx*sy*sz; h_ang_(5,0)= cx*cz-sx*sy*sz; // dR/drdy (y,z components)
        h_ang_(6,1)=-cy*cz;          h_ang_(7,1)=-sx*sy*cz;       // dR/dpdp (x,y components)
        h_ang_(8,1)= cx*sy*cz;                                    // dR/dpdp (z component)
        h_ang_(9,1)= sy*sz;          h_ang_(10,1)=-sx*cy*sz;      // dR/dpdy (x,y components)
        h_ang_(11,1)= cx*cy*sz;                                    // dR/dpdy (z component)
        h_ang_(12,2)=-cy*cz;         h_ang_(13,2)=-cx*sz-sx*sy*cz;// dR/dydy (x,y components)
        h_ang_(14,2)=-sx*sz+cx*sy*cz;                              // dR/dydy (z component)
    }
}


// --- computePointDerivatives ---
template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::computePointDerivatives(
    const Eigen::Vector3d& x,                     // Original point coordinates
    Eigen::Matrix<float, 4, 6>& point_gradient_, // Output Jacobian Jp (4x6)
    Eigen::Matrix<float, 24, 6>& point_hessian_, // Output Hessian Hp (flattened 24x6)
    bool compute_hessian)
{
    Eigen::Vector4f x4(static_cast<float>(x[0]), static_cast<float>(x[1]), static_cast<float>(x[2]), 0.0f); 

    // --- Jacobian Calculation ---
    point_gradient_.setZero();
    point_gradient_.block<3, 3>(0, 0).setIdentity(); // Derivative w.r.t translation

    Eigen::Matrix<float, 8, 1> x_j_ang = j_ang_ * x4; // (8x4) * (4x1) -> (8x1)

    point_gradient_(0, 3) = x_j_ang[0]; point_gradient_(1, 3) = x_j_ang[1]; point_gradient_(2, 3) = x_j_ang[2];
    point_gradient_(0, 4) = x_j_ang[3]; point_gradient_(1, 4) = x_j_ang[4]; point_gradient_(2, 4) = x_j_ang[5];
    point_gradient_(0, 5) = x_j_ang[6]; point_gradient_(1, 5) = x_j_ang[7];

    // --- Hessian Calculation ---
    if (compute_hessian) {
        point_hessian_.setZero(); 
        Eigen::Matrix<float, 16, 1> x_h_ang = h_ang_ * x4; // (16x4) * (4x1) -> (16x1)

        // H_rr (i=3, j=3)
        point_hessian_.block<4, 1>(3 * 4, 3) = Eigen::Vector4f(0.0f, x_h_ang[0], x_h_ang[1], 0.0f); 
        // H_rp (i=3, j=4)
        point_hessian_.block<4, 1>(3 * 4, 4) = Eigen::Vector4f(0.0f, x_h_ang[2], x_h_ang[3], 0.0f); 
        // H_ry (i=3, j=5)
        point_hessian_.block<4, 1>(3 * 4, 5) = Eigen::Vector4f(0.0f, x_h_ang[4], x_h_ang[5], 0.0f); 

        // H_pp (i=4, j=4)
        point_hessian_.block<4, 1>(4 * 4, 4) = Eigen::Vector4f(x_h_ang[6], x_h_ang[7], x_h_ang[8], 0.0f); 
        // H_py (i=4, j=5)
        point_hessian_.block<4, 1>(4 * 4, 5) = Eigen::Vector4f(x_h_ang[9], x_h_ang[10], x_h_ang[11], 0.0f); 

        // H_yy (i=5, j=5)
        point_hessian_.block<4, 1>(5 * 4, 5) = Eigen::Vector4f(x_h_ang[12], x_h_ang[13], x_h_ang[14], 0.0f); 

        // Fill symmetric blocks
        point_hessian_.block<4, 1>(4 * 4, 3) = point_hessian_.block<4, 1>(3 * 4, 4); // H_pr = H_rp
        point_hessian_.block<4, 1>(5 * 4, 3) = point_hessian_.block<4, 1>(3 * 4, 5); // H_yr = H_ry
        point_hessian_.block<4, 1>(5 * 4, 4) = point_hessian_.block<4, 1>(4 * 4, 5); // H_yp = H_py
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
    bool print_debug) 
{
    Eigen::Matrix<float, 1, 4> x_trans4(static_cast<float>(x_trans[0]), static_cast<float>(x_trans[1]), static_cast<float>(x_trans[2]), 0.0f);
    Eigen::Matrix4f c_inv4 = Eigen::Matrix4f::Zero();
    c_inv4.topLeftCorner(3, 3) = c_inv.cast<float>();

    double mahal_sq = x_trans.dot(c_inv * x_trans);

    const double max_exponent_arg = 50.0;
    if (!std::isfinite(mahal_sq) || mahal_sq < -1e-9 || (gauss_d2_ * mahal_sq > max_exponent_arg)) { 
        return 0.0; // Return zero score
    }
     if (mahal_sq < 0.0) mahal_sq = 0.0;


    double exp_term = std::exp(-gauss_d2_ * mahal_sq * 0.5);
    double score_inc = -gauss_d1_ * exp_term;
    double factor = gauss_d1_ * gauss_d2_ * exp_term;

    if (!std::isfinite(factor)) {
        return 0.0; // Return zero score
    }

    // --- Gradient Calculation ---
    Eigen::Matrix<float, 4, 6> temp_vec = c_inv4 * point_gradient4; // C^-1 * Jp (4x6)
    Eigen::Matrix<float, 1, 6> grad_contrib_float = x_trans4 * temp_vec; // (x-mu)^T * C^-1 * Jp (1x6)
    Vector6d grad_inc = factor * grad_contrib_float.transpose().cast<double>();

    if (print_debug) {
         std::cout << std::fixed << std::setprecision(5);
         std::cout << "      updateDeriv [DBG]: mahal^2=" << mahal_sq
                   << ", exp_t=" << exp_term
                   << ", factor=" << factor << std::endl;
         std::cout << "                     : grad_inc.n=" << grad_inc.norm() << std::endl;
    }

    if (!grad_inc.allFinite()){
        return 0.0; // Don't add invalid gradient
    }
    score_gradient += grad_inc;


    // --- Hessian Calculation (Approximation) ---
    if (compute_hessian) {
        Matrix6d hess_contrib = Matrix6d::Zero(); 
        Eigen::Matrix<double, 1, 6> grad_contrib_double = grad_contrib_float.cast<double>();

        // Term 1: -d2 * [grad_contrib]^T * [grad_contrib]
        hess_contrib = -gauss_d2_ * (grad_contrib_double.transpose() * grad_contrib_double);

        // Term 2: Jp^T * C^-1 * Jp
        hess_contrib += (point_gradient4.transpose() * temp_vec).cast<double>(); // temp_vec = C^-1 * Jp

        // Term 3: (x-mu)^T * C^-1 * Hp
        Eigen::Matrix<float, 1, 4> x_trans4_c_inv4 = x_trans4 * c_inv4; 
        Matrix6d term3 = Matrix6d::Zero();
        for (int i = 0; i < 6; ++i) { 
            for (int j = i; j < 6; ++j) { 
                 Eigen::Matrix<float, 4, 1> H_ij_x = point_hessian_.block<4, 1>(i * 4, j);
                 term3(i, j) = x_trans4_c_inv4 * H_ij_x;
            }
        }
        term3.template triangularView<Eigen::Lower>() = term3.template triangularView<Eigen::Upper>().transpose();
        hess_contrib += term3;

        // Scale total contribution by factor
        hess_contrib *= factor;

        if (hess_contrib.allFinite()){
            hessian += hess_contrib;
        } 
    }

    return score_inc; // Return the score contribution
}


// --- computeParticleDerivatives ---
template <typename PointSource, typename PointTarget>
double SvnNormalDistributionsTransform<PointSource, PointTarget>::computeParticleDerivatives(
    Vector6d& score_gradient, // Output gradient [x,y,z,r,p,y]
    Matrix6d& hessian,        // Output Hessian [x,y,z,r,p,y]
    const PointCloudSource& trans_cloud, // Transformed source cloud
    const Vector6d& p,        // Current pose estimate [x,y,z,r,p,y]
    bool compute_hessian)
{
    score_gradient.setZero();
    hessian.setZero();
    double total_score = 0.0;

    // Precompute Angle Derivatives (populates member j_ang_ and h_ang_)
    computeAngleDerivatives(p, compute_hessian);

    Eigen::Matrix<float, 4, 6> point_gradient4; 
    Eigen::Matrix<float, 24, 6> point_hessian24; 
    std::vector<LeafConstPtr> neighborhood; 
    std::vector<float> distances; 
    constexpr size_t reserve_size = 27; 
    neighborhood.reserve(reserve_size);
    distances.reserve(reserve_size);

    // --- Loop Over Transformed Source Points ---
    for (size_t idx = 0; idx < trans_cloud.points.size(); ++idx)
    {
        const PointSource& x_trans_pt = trans_cloud.points[idx]; 
        if (!pcl::isFinite(x_trans_pt)) continue;

        // --- Neighbor Search ---
        neighborhood.clear(); 
        distances.clear();    
        int neighbors_found = 0;
        switch (search_method_)
        {
            case NeighborSearchMethod::KDTREE:
                neighbors_found = target_cells_.radiusSearch(x_trans_pt, resolution_, neighborhood, distances);
                break;
            case NeighborSearchMethod::DIRECT7:
                neighbors_found = target_cells_.getNeighborhoodAtPoint7(x_trans_pt, neighborhood);
                break;
            case NeighborSearchMethod::DIRECT1:
            default:
                neighbors_found = target_cells_.getNeighborhoodAtPoint1(x_trans_pt, neighborhood);
                break;
        }
        if (neighbors_found == 0) continue;

        if (!input_ || idx >= input_->size()) {
             PCL_ERROR("[SvnNdt::computeParticleDerivatives] Internal error: input_ cloud invalid or index out of bounds (%zu).\n", idx);
             continue; 
        }
        const PointSource& x_pt = (*input_)[idx]; // Original point
        Eigen::Vector3d x_orig(x_pt.x, x_pt.y, x_pt.z);

        // Compute Point Derivatives (w.r.t. pose p)
        computePointDerivatives(x_orig, point_gradient4, point_hessian24, compute_hessian);

        // --- Accumulate Contributions from Neighbors ---
        double point_score_contribution = 0.0;
        Vector6d point_gradient_contribution = Vector6d::Zero();
        Matrix6d point_hessian_contribution = Matrix6d::Zero();

        for (const LeafConstPtr& cell : neighborhood)
        {
            if (!cell) continue; 
            Eigen::Vector3d x_rel = Eigen::Vector3d(x_trans_pt.x, x_trans_pt.y, x_trans_pt.z) - cell->getMean();
            const Eigen::Matrix3d& c_inv = cell->getInverseCov();

            double score_inc = updateDerivatives(point_gradient_contribution, point_hessian_contribution,
                                                 point_gradient4, point_hessian24,
                                                 x_rel, c_inv, compute_hessian,
                                                 false); // Debug print disabled
            point_score_contribution += score_inc;

        } // End loop over neighbors

        // --- Add Point's Total Contribution ---
        if (std::isfinite(point_score_contribution) &&
            point_gradient_contribution.allFinite() &&
            (!compute_hessian || point_hessian_contribution.allFinite()))
        {
            total_score += point_score_contribution;
            score_gradient += point_gradient_contribution;
            if (compute_hessian) {
                hessian += point_hessian_contribution;
            }
        } 
    } // End loop over points

    // --- Hessian Regularization (Levenberg-Marquardt style) ---
    if (compute_hessian) {
        constexpr double lambda = 1e-3; 
        hessian += lambda * Matrix6d::Identity();
    }

    if (!score_gradient.allFinite()) {
        PCL_ERROR("[SvnNdt::computeParticleDerivatives] Final score_gradient contains NaN/Inf! Resetting to zero.\n");
        score_gradient.setZero(); 
    }
    if (compute_hessian && !hessian.allFinite()) {
        PCL_ERROR("[SvnNdt::computeParticleDerivatives] Final hessian contains NaN/Inf! Resetting to identity.\n");
        hessian = Matrix6d::Identity(); 
    }

    // Return score and derivatives (in [x,y,z,r,p,y] order)
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
    SvnNdtResult result; 

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
        result.converged = false; 
        return result;
    }

    // --- Initialization ---
    input_ = source_cloud.makeShared();

    std::vector<gtsam::Pose3> particles(K_);
    // Order: [roll, pitch, yaw, x, y, z] in radians and meters
    Vector6d initial_sigmas; initial_sigmas << 0.02, 0.02, 0.05, 0.1, 0.1, 0.1; 
    auto prior_noise_model = gtsam::noiseModel::Diagonal::Sigmas(initial_sigmas);
    gtsam::Sampler sampler(prior_noise_model, std::chrono::system_clock::now().time_since_epoch().count()); 

    for (int k = 0; k < K_; ++k) {
        particles[k] = prior_mean.retract(sampler.sample());
    }

    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> loss_gradients(K_); // In NDT order [x,y,z,r,p,y]
    std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>> loss_hessians(K_);  // In NDT order [x,y,z,r,p,y]
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> particle_updates(K_); // In GTSAM order [r,p,y,x,y,z]
    std::vector<PointCloudSource> transformed_clouds(K_); 

    Matrix6d I6 = Matrix6d::Identity(); 


    // --- BEGIN CRITICAL FIX: Define Permutation Matrix ---
    // This matrix converts from NDT tangent space [x,y,z,r,p,y] (indices 0-5)
    // to GTSAM tangent space [r,p,y,x,y,z] (indices 0-5)
    Eigen::Matrix<double, 6, 6> P_gtsam_from_ndt;
    P_gtsam_from_ndt.setZero();
    // Move NDT indices 3,4,5 (r,p,y) to GTSAM indices 0,1,2
    P_gtsam_from_ndt.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    // Move NDT indices 0,1,2 (x,y,z) to GTSAM indices 3,4,5
    P_gtsam_from_ndt.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity();
    // This matrix is symmetric, so P.transpose() == P
    // --- END CRITICAL FIX ---


    // --- SVN Iteration Loop ---
    double avg_update_norm = std::numeric_limits<double>::max(); // FIX 2: Correctly scoped
    for (int iter = 0; iter < max_iter_; ++iter)
    {
        auto iter_start_time = std::chrono::high_resolution_clock::now(); 

        // --- Stage 1: Compute NDT Derivatives (Parallel using TBB) ---
        auto stage1_start_time = std::chrono::high_resolution_clock::now();
        tbb::parallel_for(tbb::blocked_range<int>(0, K_),
            [&](const tbb::blocked_range<int>& r) {

            // Thread-local copy for thread-safe modification of j_ang_, h_ang_
            SvnNormalDistributionsTransform<PointSource, PointTarget> local_ndt = *this;
            local_ndt.input_ = this->input_; 

            Vector6d grad_k; // NDT order
            Matrix6d hess_k; // NDT order

            for (int k = r.begin(); k < r.end(); ++k) {
                pcl::transformPointCloud(source_cloud, transformed_clouds[k], particles[k].matrix().cast<float>());

                // Convert gtsam::Pose3 particle to NDT's expected [x,y,z,r,p,y] vector
                Vector6d p_k_ndt;
                gtsam::Vector3 rpy = particles[k].rotation().rpy(); 
                p_k_ndt.head<3>() = particles[k].translation(); // FIX 1: Correctly get translation
                p_k_ndt.tail<3>() = rpy; 

                // Compute NDT score, gradient, and Hessian for this particle's pose
                // grad_k and hess_k are filled in NDT order [x,y,z,r,p,y]
                double score_k = local_ndt.computeParticleDerivatives(grad_k, hess_k, transformed_clouds[k], p_k_ndt, true);

                // Store NEGATED gradient and Hessian (still in NDT order)
                loss_gradients[k] = -grad_k;
                loss_hessians[k] = -hess_k;

                 if (!hess_k.allFinite() || hess_k.hasNaN()) {
                     PCL_WARN("[SvnNdt::align Stage1] NaN/Inf in NDT Hessian for particle %d, iter %d. Using -Identity.\n", k, iter);
                     loss_hessians[k] = -I6; 
                 } else if (loss_hessians[k].hasNaN()) {
                     PCL_WARN("[SvnNdt::align Stage1] Stored loss_hessian has NaN for particle %d, iter %d. Using -Identity.\n", k, iter);
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
                Vector6d phi_k_star = Vector6d::Zero(); // SVGD direction (in GTSAM order)
                Matrix6d H_k_tilde = Matrix6d::Zero();  // SVN Hessian (in GTSAM order)

                // Aggregate contributions from all particles (l) to particle k
                for (int l = 0; l < K_; ++l) {
                    // Kernel value and gradient (operate on gtsam::Pose3)
                    // k_grad is in GTSAM order [r,p,y,x,y,z]
                    double k_val = rbf_kernel(particles[l], particles[k]);
                    Vector6d k_grad = rbf_kernel_gradient(particles[l], particles[k]); 

                    if (!std::isfinite(k_val) || !k_grad.allFinite()) {
                         PCL_WARN("[SvnNdt::align Stage2] NaN/Inf in kernel computation between particles %d and %d, iter %d.\n", l, k, iter);
                         continue; 
                     }

                    // --- BEGIN CRITICAL FIX: Permute NDT derivatives to GTSAM order ---
                    Vector6d grad_l_gtsam_order = P_gtsam_from_ndt * loss_gradients[l];
                    // H_gtsam = P * H_ndt * P^T. (P is symmetric, so P^T = P)
                    Matrix6d hess_l_gtsam_order = P_gtsam_from_ndt * loss_hessians[l] * P_gtsam_from_ndt;
                    // --- END CRITICAL FIX ---

                    // Accumulate SVGD direction term: k(l,k)*grad(loss_l) + grad_l(k(l,k))
                    if (!loss_gradients[l].allFinite()) { // Check original
                         PCL_WARN("[SvnNdt::align Stage2] NaN/Inf in loss_gradient for particle %d, iter %d. Skipping term in phi_k_star.\n", l, iter);
                    } else {
                         phi_k_star += k_val * grad_l_gtsam_order; // <-- USE PERMUTED GRADIENT
                    }
                    phi_k_star += k_grad; // Always add kernel gradient (already GTSAM order)


                    // Accumulate SVN Hessian term: k(l,k)^2 * hess(loss_l) + grad_l(k(l,k)) * grad_l(k(l,k))^T
                    if (loss_hessians[l].allFinite()) { // Check original
                        H_k_tilde += (k_val * k_val) * hess_l_gtsam_order; // <-- USE PERMUTED HESSIAN
                    } 
                    // Always add the kernel gradient term (already GTSAM order)
                    H_k_tilde += (k_grad * k_grad.transpose());
                }

                // Average over particles
                if (K_ > 0) {
                    phi_k_star /= static_cast<double>(K_);
                    H_k_tilde /= static_cast<double>(K_);
                }

                // Add regularization to the SVN Hessian for stability
                constexpr double svn_hess_lambda = 1e-4;
                H_k_tilde += svn_hess_lambda * I6;

                // Solve the linear system: H_k_tilde * update = phi_k_star
                // All terms are now in GTSAM order, so the resulting 'update'
                // will also be in GTSAM order [r,p,y,x,y,z].
                Eigen::LDLT<Matrix6d> solver(H_k_tilde);
                if (solver.info() == Eigen::Success && H_k_tilde.allFinite()) {
                     Vector6d update = solver.solve(phi_k_star);
                     if (update.allFinite()) {
                         particle_updates[k] = update; // Store the [r,p,y,x,y,z] update
                     } else {
                         PCL_WARN("[SvnNdt::align Stage2] Solver produced NaN/Inf update for particle %d, iter %d. Setting update to zero.\n", k, iter);
                         particle_updates[k].setZero();
                     }
                } else {
                    PCL_ERROR("[SvnNdt::align Stage2] LDLT solver failed or H_tilde invalid for particle %d, iter %d (Info: %d). Setting update to zero.\n", k, iter, solver.info());
                    particle_updates[k].setZero();
                }
            }
        }); // End TBB Stage 2
        auto stage2_end_time = std::chrono::high_resolution_clock::now();

        // --- Stage 3: Apply Updates to Particles (Serial) ---
        double total_update_norm_sq = 0.0; 
        for (int k = 0; k < K_; ++k) {
            // particle_updates[k] is in GTSAM order [r,p,y,x,y,z]
            Vector6d scaled_update = step_size_ * particle_updates[k]; 

             if (!scaled_update.allFinite()) {
                 PCL_WARN("[SvnNdt::align Stage3] NaN/Inf in scaled update for particle %d, iter %d. Skipping update.\n", k, iter);
                 continue; 
             }

            total_update_norm_sq += particle_updates[k].squaredNorm();

            // Apply the update on the manifold
            // gtsam::Pose3::retract correctly expects a [r,p,y,x,y,z] tangent vector
            particles[k] = particles[k].retract(scaled_update);
        }
        auto stage3_end_time = std::chrono::high_resolution_clock::now();

        // --- Check Convergence ---
        result.iterations = iter + 1;
        avg_update_norm = (K_ > 0) ? std::sqrt(total_update_norm_sq / static_cast<double>(K_)) : 0.0;

        auto iter_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> stage1_ms = stage1_end_time - stage1_start_time;
        std::chrono::duration<double, std::milli> stage2_ms = stage2_end_time - stage2_start_time;
        std::chrono::duration<double, std::milli> stage3_ms = stage3_end_time - stage2_end_time; 
        std::chrono::duration<double, std::milli> iter_ms = iter_end_time - iter_start_time;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "[SVN Iter " << std::setw(2) << iter << "] Avg Update Norm: " << avg_update_norm
                  << " (T: " << std::setprecision(1) << iter_ms.count() << "ms = "
                  << "S1:" << stage1_ms.count() << " + S2:" << stage2_ms.count() << " + S3:" << stage3_ms.count() << ")" << std::endl;

        if (avg_update_norm < stop_thresh_) {
            result.converged = true;
            PCL_WARN_STREAM("[SvnNdt::align] Converged in " << result.iterations
                 << " iterations (Avg Update Norm: " << std::fixed << std::setprecision(6) << avg_update_norm
                 << " < " << stop_thresh_ << ").\n");
            break; 
        }

    } // End SVN iteration loop

    // --- Finalization: Compute Mean Pose and Covariance from Particles ---
    // This part was already correct as it uses only GTSAM functions.
    if (K_ > 0) {
        // --- Calculate Mean Pose ---
        Vector6d mean_xi_at_prior = Vector6d::Zero();
        for(int k=0; k < K_; ++k) {
            mean_xi_at_prior += gtsam::Pose3::Logmap(prior_mean.between(particles[k]));
        }
        mean_xi_at_prior /= static_cast<double>(K_);
        result.final_pose = prior_mean.retract(mean_xi_at_prior);

        // --- Calculate Covariance ---
        if (K_ > 1) {
            result.final_covariance.setZero();
            std::vector<Vector6d> tangent_vectors_at_mean(K_);
             Vector6d mean_xi_at_mean = Vector6d::Zero(); 
             for(int k=0; k < K_; ++k) {
                 tangent_vectors_at_mean[k] = gtsam::Pose3::Logmap(result.final_pose.between(particles[k]));
                 mean_xi_at_mean += tangent_vectors_at_mean[k];
             }
             mean_xi_at_mean /= static_cast<double>(K_); 

            // Compute sample covariance in the tangent space at the mean pose
            for(int k=0; k < K_; ++k) {
                Vector6d diff = tangent_vectors_at_mean[k] - mean_xi_at_mean; 
                result.final_covariance += diff * diff.transpose();
            }
            result.final_covariance /= static_cast<double>(K_ - 1); // N-1
        } else {
            result.final_covariance = prior_noise_model->covariance(); 
        }

        // --- Final Covariance Regularization ---
        Eigen::SelfAdjointEigenSolver<Matrix6d> final_eigensolver(result.final_covariance);
        if (final_eigensolver.info() == Eigen::Success) {
            Vector6d final_evals = final_eigensolver.eigenvalues();
            if (final_evals(0) < 1e-9) { 
                 PCL_DEBUG("[SvnNdt::align] Final covariance has small/negative eigenvalues. Applying regularization.\n");
                 for(int i=0; i<6; ++i) final_evals(i) = std::max(final_evals(i), 1e-9);
                 result.final_covariance = final_eigensolver.eigenvectors() * final_evals.asDiagonal() * final_eigensolver.eigenvectors().transpose();
            }
        } else {
             PCL_WARN("[SvnNdt::align] Eigendecomposition failed for final covariance. Using regularized identity.\n");
             result.final_covariance = 1e-6 * I6; 
        }

    } else { 
        result.final_pose = prior_mean;
        result.final_covariance.setIdentity(); 
    }

    input_.reset();

    if (!result.converged && result.iterations >= max_iter_) {
        // FIX 2: Correctly accesses avg_update_norm
        PCL_WARN_STREAM("[SvnNdt::align] Reached max iterations (" << max_iter_
                      << ") without converging (Avg Update Norm: " << std::fixed << std::setprecision(6) << avg_update_norm
                      << " >= " << stop_thresh_ << ").\n");
    }

    return result;
}

} // namespace svn_ndt

#endif // SVN_NDT_SVN_NDT_IMPL_HPP_