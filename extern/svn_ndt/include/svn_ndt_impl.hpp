#ifndef SVN_NDT_SVN_NDT_IMPL_HPP_
#define SVN_NDT_SVN_NDT_IMPL_HPP_

// Include the class header
#include <svn_ndt.h>

// --- Standard/External Library Includes ---
#include <vector>
#include <cmath>
#include <numeric> // For std::accumulate
#include <limits>
#include <iostream> // For PCL_WARN/ERROR and debug output
#include <chrono>   // For timing if needed

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

// GTSAM for pose representation and operations
#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/numericalDerivative.h> // Not directly used, but good dependency check
#include <gtsam/inference/Symbol.h> // For Symbol if used in testing/debugging
#include <gtsam/nonlinear/Values.h> // For Values if used in testing/debugging
#include <gtsam/nonlinear/NonlinearFactorGraph.h> // If adding factors directly

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
      search_method_(NeighborSearchMethod::DIRECT1), // Default search method
      K_(30), max_iter_(50), kernel_h_(0.1), step_size_(0.1), stop_thresh_(1e-4)
{
    updateNdtConstants(); // Initialize gauss_d* constants
    j_ang_.setZero();
    h_ang_.setZero();
}

template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::updateNdtConstants()
{
    // Recalculate NDT constants based on current resolution and outlier ratio
    // (Copied from ndt_omp constructor logic)
    if (resolution_ <= 0) return; // Avoid division by zero

    double gauss_c1 = 10.0 * (1.0 - outlier_ratio_);
    double gauss_c2 = outlier_ratio_ / pow(resolution_, 3);

    // Prevent log(0) or log(negative)
    if (gauss_c1 <= 1e-9) gauss_c1 = 1e-9;
    if (gauss_c2 <= 1e-9) gauss_c2 = 1e-9;
    double c1_plus_c2 = gauss_c1 + gauss_c2;
    if (c1_plus_c2 <= 1e-9) c1_plus_c2 = 1e-9;
    double intermediate_log_arg = gauss_c1 * exp(-0.5) + gauss_c2;
     if (intermediate_log_arg <= 1e-9) intermediate_log_arg = 1e-9;


    gauss_d3_ = -log(gauss_c2);
    gauss_d1_ = -log(c1_plus_c2) - gauss_d3_;

    double term_for_d2_log = (-log(intermediate_log_arg) - gauss_d3_) / gauss_d1_;
    // Check for invalid arguments to log, specifically term_for_d2_log <= 0
    if (term_for_d2_log <= 1e-9) {
        PCL_WARN("[SvnNdt] Invalid argument for log in gauss_d2 calculation. Using default.\n");
        // Assign a default or handle error appropriately. Using a typical value.
        gauss_d2_ = 2.0; // Corresponds to approx exp(-1 * mahal^2) term
    } else {
        gauss_d2_ = -2.0 * log(term_for_d2_log);
    }


    // Final check for NaN
    if (std::isnan(gauss_d1_) || std::isnan(gauss_d2_) || std::isnan(gauss_d3_)) {
        PCL_WARN("[SvnNdt] NaN detected in NDT constant calculation. Check resolution (%.3f) and outlier ratio (%.3f).\n", resolution_, outlier_ratio_);
        // Set safe defaults
        gauss_d1_ = 1.0; gauss_d2_ = 1.0; gauss_d3_ = 0.0;
    }
    // Debug print
    // std::cout << "NDT Constants: d1=" << gauss_d1_ << ", d2=" << gauss_d2_ << ", d3=" << gauss_d3_ << std::endl;
}

//=================================================================================================
// Configuration
//=================================================================================================
template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::setInputTarget(
    const PointCloudTargetConstPtr& cloud)
{
    if (!cloud || cloud->empty()) {
        PCL_ERROR("[SvnNdt] Invalid or empty target point cloud provided.\n");
        target_cells_.setInputCloud(nullptr); // Clear previous target
        return;
    }
    target_cells_.setInputCloud(cloud);

    if (resolution_ > 0) {
        target_cells_.setLeafSize(resolution_, resolution_, resolution_);
        bool build_kdtree = (search_method_ == NeighborSearchMethod::KDTREE);
        target_cells_.filter(build_kdtree); // Build grid & optionally kdtree
        updateNdtConstants(); // Update constants based on resolution
    } else {
        PCL_WARN("[SvnNdt] Target set, but resolution is not positive. Grid not built.\n");
    }
}

template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::setResolution(float resolution)
{
    if (resolution <= 0) {
        PCL_ERROR("[SvnNdt] Resolution must be positive.\n");
        return;
    }
    resolution_ = resolution;
    // If target already exists, rebuild the grid and update constants
    if (target_cells_.getInputCloud()) {
        setInputTarget(target_cells_.getInputCloud());
    }
}

template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::setOutlierRatio(double ratio)
{
    if (ratio < 0.0 || ratio >= 1.0) {
         PCL_WARN("[SvnNdt] Outlier ratio must be in [0, 1). Clamping to range.\n");
         outlier_ratio_ = std::max(0.0, std::min(ratio, 0.999));
    } else {
        outlier_ratio_ = ratio;
    }
    updateNdtConstants(); // Recalculate gauss_d* constants
}


//=================================================================================================
// RBF Kernel Functions (Implementations)
//=================================================================================================
template <typename PointSource, typename PointTarget>
double SvnNormalDistributionsTransform<PointSource, PointTarget>::rbf_kernel(
    const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const
{
    if (kernel_h_ <= 1e-9) return 1.0; // Avoid division by zero, return max value
    Vector6d diff_log = gtsam::Pose3::Logmap(pose_l.between(pose_k));
    double sq_norm = diff_log.squaredNorm();
    return std::exp(-sq_norm / kernel_h_);
}

template <typename PointSource, typename PointTarget>
typename SvnNormalDistributionsTransform<PointSource, PointTarget>::Vector6d
SvnNormalDistributionsTransform<PointSource, PointTarget>::rbf_kernel_gradient(
    const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const
{
     if (kernel_h_ <= 1e-9) return Vector6d::Zero(); // Avoid division by zero
    Vector6d diff_log = gtsam::Pose3::Logmap(pose_l.between(pose_k));
    double sq_norm = diff_log.squaredNorm();
    double k_val = std::exp(-sq_norm / kernel_h_);

    // Use the tangent space difference approximation for the gradient w.r.t. first argument
    return k_val * (-2.0 / kernel_h_) * diff_log;
}


//=================================================================================================
// NDT Math Functions (Adapted SERIAL Implementations)
//=================================================================================================

// --- computeAngleDerivatives ---
// [Copied Verbatim from Previous Response - Assumed Correct]
template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::computeAngleDerivatives(
    const Vector6d& p, bool compute_hessian)
{
    // Simplified math for near 0 angles
    double cx, cy, cz, sx, sy, sz;
    constexpr double epsilon = 1e-5;
    if (std::abs(p(3)) < epsilon) { sx = 0.0; cx = 1.0; } else { sx = sin(p(3)); cx = cos(p(3)); }
    if (std::abs(p(4)) < epsilon) { sy = 0.0; cy = 1.0; } else { sy = sin(p(4)); cy = cos(p(4)); }
    if (std::abs(p(5)) < epsilon) { sz = 0.0; cz = 1.0; } else { sz = sin(p(5)); cz = cos(p(5)); }

    j_ang_.setZero();
    j_ang_(0,0)=-sx*sz+cx*sy*cz; j_ang_(0,1)=-sx*cz-cx*sy*sz; j_ang_(0,2)=-cx*cy;
    j_ang_(1,0)= cx*sz+sx*sy*cz; j_ang_(1,1)= cx*cz-sx*sy*sz; j_ang_(1,2)=-sx*cy;
    j_ang_(2,0)=-sy*cz;          j_ang_(2,1)= sy*sz;          j_ang_(2,2)= cy;
    j_ang_(3,0)= sx*cy*cz;       j_ang_(3,1)=-sx*cy*sz;       j_ang_(3,2)= sx*sy;
    j_ang_(4,0)=-cx*cy*cz;       j_ang_(4,1)= cx*cy*sz;       j_ang_(4,2)=-cx*sy;
    j_ang_(5,0)=-cy*sz;          j_ang_(5,1)=-cy*cz;          j_ang_(5,2)= 0.0;
    j_ang_(6,0)= cx*cz-sx*sy*sz; j_ang_(6,1)=-cx*sz-sx*sy*cz; j_ang_(6,2)= 0.0;
    j_ang_(7,0)= sx*cz+cx*sy*sz; j_ang_(7,1)= cx*sy*cz-sx*sz; j_ang_(7,2)= 0.0;

    if (compute_hessian) {
        h_ang_.setZero();
        h_ang_(0,0)=-cx*sz-sx*sy*cz; h_ang_(0,1)=-cx*cz+sx*sy*sz; h_ang_(0,2)= sx*cy;
        h_ang_(1,0)=-sx*sz+cx*sy*cz; h_ang_(1,1)=-cx*sy*sz-sx*cz; h_ang_(1,2)=-cx*cy;
        h_ang_(2,0)= cx*cy*cz;       h_ang_(2,1)=-cx*cy*sz;       h_ang_(2,2)= cx*sy;
        h_ang_(3,0)= sx*cy*cz;       h_ang_(3,1)=-sx*cy*sz;       h_ang_(3,2)= sx*sy;
        h_ang_(4,0)=-sx*cz-cx*sy*sz; h_ang_(4,1)= sx*sz-cx*sy*cz; h_ang_(4,2)= 0.0;
        h_ang_(5,0)= cx*cz-sx*sy*sz; h_ang_(5,1)=-sx*sy*cz-cx*sz; h_ang_(5,2)= 0.0;
        h_ang_(6,0)=-cy*cz;          h_ang_(6,1)= cy*sz;          h_ang_(6,2)= sy;
        h_ang_(7,0)=-sx*sy*cz;       h_ang_(7,1)= sx*sy*sz;       h_ang_(7,2)= sx*cy;
        h_ang_(8,0)= cx*sy*cz;       h_ang_(8,1)=-cx*sy*sz;       h_ang_(8,2)=-cx*cy;
        h_ang_(9,0)= sy*sz;          h_ang_(9,1)= sy*cz;          h_ang_(9,2)= 0.0;
        h_ang_(10,0)=-sx*cy*sz;      h_ang_(10,1)=-sx*cy*cz;      h_ang_(10,2)= 0.0;
        h_ang_(11,0)= cx*cy*sz;      h_ang_(11,1)= cx*cy*cz;      h_ang_(11,2)= 0.0;
        h_ang_(12,0)=-cy*cz;         h_ang_(12,1)= cy*sz;         h_ang_(12,2)= 0.0;
        h_ang_(13,0)=-cx*sz-sx*sy*cz;h_ang_(13,1)=-cx*cz+sx*sy*sz;h_ang_(13,2)= 0.0;
        h_ang_(14,0)=-sx*sz+cx*sy*cz;h_ang_(14,1)=-cx*sy*sz-sx*cz;h_ang_(14,2)= 0.0;
    }
}

// --- computePointDerivatives ---
// [Copied Verbatim from Previous Response - Assumed Correct]
template <typename PointSource, typename PointTarget>
void SvnNormalDistributionsTransform<PointSource, PointTarget>::computePointDerivatives(
    const Eigen::Vector3d& x,
    Eigen::Matrix<float, 4, 6>& point_gradient_,
    Eigen::Matrix<float, 24, 6>& point_hessian_,
    bool compute_hessian)
{
    Eigen::Vector4f x4(static_cast<float>(x[0]), static_cast<float>(x[1]), static_cast<float>(x[2]), 0.0f);
    point_gradient_.setZero(); // Clear previous point's data
    point_gradient_.block<3, 3>(0, 0).setIdentity();

    Eigen::Matrix<float, 8, 1> x_j_ang = j_ang_ * x4;

    point_gradient_(0, 3) = x_j_ang[0]; point_gradient_(1, 3) = x_j_ang[1]; point_gradient_(2, 3) = x_j_ang[2];
    point_gradient_(0, 4) = x_j_ang[3]; point_gradient_(1, 4) = x_j_ang[4]; point_gradient_(2, 4) = x_j_ang[5];
    point_gradient_(0, 5) = x_j_ang[6]; point_gradient_(1, 5) = x_j_ang[7];

    if (compute_hessian) {
        point_hessian_.setZero(); // Clear previous point's data
        Eigen::Matrix<float, 16, 1> x_h_ang = h_ang_ * x4;

        // Mapping based on ndt_omp_impl.hpp:544-577, using 4x1 blocks
        // Indices: p = [t1,t2,t3, r1,r2,r3] -> idx 0..5. NDT uses r1=3, r2=4, r3=5
        // H_11 -> block (3*4, 3) = (12, 3)
        point_hessian_.block<4, 1>(12, 3) = Eigen::Vector4f(0.0f, x_h_ang[0], x_h_ang[1], 0.0f); // a2, a3
        // H_12 -> block (3*4, 4) = (12, 4)
        point_hessian_.block<4, 1>(12, 4) = Eigen::Vector4f(0.0f, x_h_ang[2], x_h_ang[3], 0.0f); // b2, b3
        // H_13 -> block (3*4, 5) = (12, 5)
        point_hessian_.block<4, 1>(12, 5) = Eigen::Vector4f(0.0f, x_h_ang[4], x_h_ang[5], 0.0f); // c2, c3
        // H_22 -> block (4*4, 4) = (16, 4)
        point_hessian_.block<4, 1>(16, 4) = Eigen::Vector4f(x_h_ang[6], x_h_ang[7], x_h_ang[8], 0.0f); // d1, d2, d3
        // H_23 -> block (4*4, 5) = (16, 5)
        point_hessian_.block<4, 1>(16, 5) = Eigen::Vector4f(x_h_ang[9], x_h_ang[10], x_h_ang[11], 0.0f); // e1, e2, e3
        // H_33 -> block (5*4, 5) = (20, 5)
        point_hessian_.block<4, 1>(20, 5) = Eigen::Vector4f(x_h_ang[12], x_h_ang[13], x_h_ang[14], 0.0f); // f1, f2, f3

        // Fill symmetric parts
        point_hessian_.block<4, 1>(16, 3) = point_hessian_.block<4, 1>(12, 4); // H_21 = H_12
        point_hessian_.block<4, 1>(20, 3) = point_hessian_.block<4, 1>(12, 5); // H_31 = H_13
        point_hessian_.block<4, 1>(20, 4) = point_hessian_.block<4, 1>(16, 5); // H_32 = H_23
    }
}


// --- updateDerivatives ---
// [Copied Verbatim from Previous Response - Assumed Correct, added finite check]
template <typename PointSource, typename PointTarget>
double SvnNormalDistributionsTransform<PointSource, PointTarget>::updateDerivatives(
    Vector6d& score_gradient, Matrix6d& hessian,
    const Eigen::Matrix<float, 4, 6>& point_gradient4,
    const Eigen::Matrix<float, 24, 6>& point_hessian_,
    const Eigen::Vector3d& x_trans, const Eigen::Matrix3d& c_inv,
    bool compute_hessian, bool print_debug)
{
    Eigen::Matrix<float, 1, 4> x_trans4(static_cast<float>(x_trans[0]), static_cast<float>(x_trans[1]), static_cast<float>(x_trans[2]), 0.0f);
    Eigen::Matrix4f c_inv4 = Eigen::Matrix4f::Zero();
    c_inv4.topLeftCorner(3, 3) = c_inv.cast<float>();

    double mahal_sq = x_trans.dot(c_inv * x_trans);
    // Protect against exp(-inf) or exp(large positive) if mahal_sq is NaN or huge neg
    if (!std::isfinite(mahal_sq) || (gauss_d2_ * mahal_sq > 50.0)) { // exp(-25) is already tiny
        // ######### DEBUG
        // if (print_debug) {
        //      std::cout << "      updateDeriv[pt0]: mahal_sq invalid/large: " << mahal_sq << std::endl;
        // }
        // ######### DEBUG
        return 0.0;
    }
    double exp_term = std::exp(-gauss_d2_ * mahal_sq * 0.5);
    double score_inc = -gauss_d1_ * exp_term;
    double factor = gauss_d1_ * gauss_d2_ * exp_term; // == -gauss_d2_ * score_inc;

    if (!std::isfinite(factor)) { // <-- REMOVE 'factor < 0' CHECK
        // ######### DEBUG
        // if (print_debug) {
        //     std::cout << "      updateDeriv[pt0]: factor NON-FINITE: " << factor << std::endl; // <-- Update message
        // }
        // ######### DEBUG
        return 0.0; // Still return 0 if NaN/Inf
    }

    Eigen::Matrix<float, 4, 6> temp_vec = c_inv4 * point_gradient4;
    Eigen::Matrix<float, 1, 6> grad_contrib_float = x_trans4 * temp_vec;
    Vector6d grad_inc = factor * grad_contrib_float.transpose().cast<double>();
    score_gradient += grad_inc;
    // score_gradient += factor * grad_contrib_float.transpose().cast<double>();

    // ######### DEBUG
    // if (print_debug) {
    //      std::cout << "      updateDeriv[pt0]: mahal_sq=" << mahal_sq 
    //                << ", exp_term=" << exp_term 
    //                << ", factor=" << factor 
    //                << ", grad_contrib_norm=" << grad_contrib_float.norm()
    //                << ", grad_inc_norm=" << grad_inc.norm() << std::endl;
    // }
    // ######### DEBUG

    if (compute_hessian) {
        Eigen::Matrix<double, 6, 6> hess_contrib = Matrix6d::Zero(); // Accumulate contribution here
        Eigen::Matrix<double, 1, 6> grad_contrib_double = grad_contrib_float.cast<double>();

        // Term 1: -d2 * grad * grad^T
        hess_contrib = -gauss_d2_ * (grad_contrib_double.transpose() * grad_contrib_double);

        // Term 2: Jp^T * C^-1 * Jp
        hess_contrib += (point_gradient4.transpose() * temp_vec).cast<double>();

        // Term 3: (x-mu)^T * C^-1 * Hp
        Eigen::Matrix<float, 1, 4> x_trans4_c_inv4 = x_trans4 * c_inv4;
        Matrix6d term3 = Matrix6d::Zero();
        for (int i = 0; i < 6; ++i) {
            for (int j = i; j < 6; ++j) { // Use symmetry
                 // Extract H_ij block (4x1) - Use mapping from computePointDerivatives
                 // Block for H_ij is expected at (i*4, j) ? Let's try direct access based on original
                 Eigen::Matrix<float, 4, 1> H_ij_x = point_hessian_.block<4, 1>(i * 4, j);
                 term3(i, j) = x_trans4_c_inv4 * H_ij_x;
            }
        }
        term3.template triangularView<Eigen::Lower>() = term3.template triangularView<Eigen::Upper>().transpose();
        hess_contrib += term3;

        hessian += factor * hess_contrib;
    }
    return score_inc;
}


// --- computeParticleDerivatives ---
// [Updated to include neighbor search switch]
template <typename PointSource, typename PointTarget>
double SvnNormalDistributionsTransform<PointSource, PointTarget>::computeParticleDerivatives(
    Vector6d& score_gradient, Matrix6d& hessian,
    const PointCloudSource& trans_cloud, const Vector6d& p,
    bool compute_hessian)
{
    // --- Initializations ---
    score_gradient.setZero();
    hessian.setZero();
    double total_score = 0.0;
    bool first_point_processed = false; // For debug prints in updateDerivatives

    // --- Precompute Angle Derivatives ---
    // (This modifies member variables j_ang_ and h_ang_ of the 'local_ndt' copy)
    computeAngleDerivatives(p, compute_hessian);

    // --- Reusable Structures ---
    Eigen::Matrix<float, 4, 6> point_gradient4; point_gradient4.setZero();
    Eigen::Matrix<float, 24, 6> point_hessian24; point_hessian24.setZero();
    std::vector<LeafConstPtr> neighborhood;
    neighborhood.reserve(27); // Max expected neighbors
    std::vector<float> distances;
    distances.reserve(27);

    // --- Loop Over Transformed Source Points ---
    for (size_t idx = 0; idx < trans_cloud.points.size(); ++idx)
    {
        const PointSource& x_trans_pt = trans_cloud.points[idx];
        if (!pcl::isFinite(x_trans_pt)) continue; // Skip invalid points

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
        if (neighbors_found == 0) continue; // Skip if no valid neighbors

        // --- Get Original Point Coordinates ---
        if (!input_ || idx >= input_->size()) {
             PCL_ERROR("[SvnNdt] Internal error: input_ cloud invalid or index out of bounds during derivative calculation.\n");
             continue; // Skip this point
        }
        const PointSource& x_pt = (*input_)[idx];
        Eigen::Vector3d x_orig(x_pt.x, x_pt.y, x_pt.z);

        // --- Compute Point Derivatives (Jacobian/Hessian w.r.t. pose) ---
        // (This uses the precomputed j_ang_ / h_ang_)
        computePointDerivatives(x_orig, point_gradient4, point_hessian24, compute_hessian);

        // --- Accumulate Contributions from Neighbors ---
        double point_score_contribution = 0.0;
        Vector6d point_gradient_contribution = Vector6d::Zero();
        Matrix6d point_hessian_contribution = Matrix6d::Zero();
        bool is_first_point = !first_point_processed; // Flag for debug print

        for (const LeafConstPtr& cell : neighborhood)
        {
            if (!cell) continue; // Safety check

            Eigen::Vector3d x_rel = Eigen::Vector3d(x_trans_pt.x, x_trans_pt.y, x_trans_pt.z) - cell->getMean();
            const Eigen::Matrix3d& c_inv = cell->getInverseCov();

            // --- Core NDT derivative update ---
            double score_inc = updateDerivatives(point_gradient_contribution, point_hessian_contribution,
                                            point_gradient4, point_hessian24,
                                            x_rel, c_inv, compute_hessian, is_first_point); // Pass debug flag
            point_score_contribution += score_inc;

        } // End loop over neighbors

        // --- Update Debug Flag ---
        if (is_first_point) {
             first_point_processed = true;
        }

        // --- Add Point's Total Contribution to Overall Gradient/Hessian ---
        // (Check finiteness just in case updateDerivatives had issues despite internal checks)
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
            // PCL_WARN("[SvnNdt] NaN/Inf detected in point contribution (idx %zu). Skipping.\n", idx);
        }

    } // End loop over points

    // --- >>> ADD HESSIAN REGULARIZATION (Levenberg-Marquardt style) <<< ---
    if (compute_hessian) {
        double lambda = 0.1; // Regularization strength (tune if needed)
        hessian += lambda * Matrix6d::Identity();
    }
    // --- >>> END REGULARIZATION <<< ---

    // --- Final Sanity Checks & Debug Print ---
    if (!score_gradient.allFinite()) {
        // std::cerr << "computeParticleDerivatives: Final score_gradient has NaN/Inf!" << std::endl;
        score_gradient.setZero(); // Avoid propagating errors
    }
    if (compute_hessian && !hessian.allFinite()) {
        // std::cerr << "computeParticleDerivatives: Final hessian has NaN/Inf!" << std::endl;
        hessian = Matrix6d::Identity(); // Use identity if invalid
    }

    // --- Debug Print for Final Gradient Norm ---
    std::cout << "    computeParticleDeriv Final Grad Norm: " << score_gradient.norm() << std::endl;
    // ---

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
    SvnNdtResult result; // Store results here

    // --- Input Checks ---
    if (!target_cells_.getInputCloud() || target_cells_.getAllLeaves().empty()) {
        PCL_ERROR("[SvnNdt::align] Target grid not initialized. Call setInputTarget() first.\n");
        result.converged = false;
        return result;
    }
    if (source_cloud.empty()) {
        PCL_ERROR("[SvnNdt::align] Input source cloud is empty.\n");
        result.converged = false;
        return result;
    }
    if (K_ <= 0) {
        PCL_ERROR("[SvnNdt::align] Particle count must be positive.\n");
        result.converged = false;
        return result;
    }

    // --- Initialization ---
    input_ = source_cloud.makeShared(); // Store const ptr to source cloud

    std::vector<gtsam::Pose3> particles(K_);
    // Use a Gaussian sampler for initialization noise
    // Define initial noise sigma values (tune these!) r,p,y,x,y,z
    Vector6d initial_sigmas; initial_sigmas << 0.02, 0.02, 0.05, 0.1, 0.1, 0.1; // Example values
    auto prior_noise_model = gtsam::noiseModel::Diagonal::Sigmas(initial_sigmas);
    gtsam::Sampler sampler(prior_noise_model); // Create sampler

    for (int k = 0; k < K_; ++k) {
        particles[k] = prior_mean.retract(sampler.sample());
    }

    // Allocate storage
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> loss_gradients(K_);
    std::vector<Matrix6d, Eigen::aligned_allocator<Matrix6d>> loss_hessians(K_);
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> particle_updates(K_);
    std::vector<PointCloudSource> transformed_clouds(K_); // Use aligned allocator if PointSource has Eigen types

    Matrix6d I6 = Matrix6d::Identity(); // Reusable identity matrix

    // --- SVN Iteration Loop ---
    for (int iter = 0; iter < max_iter_; ++iter)
    {
        // --- Stage 1: Compute NDT Derivatives (Parallel using TBB) ---
        tbb::parallel_for(tbb::blocked_range<int>(0, K_),
            [&](const tbb::blocked_range<int>& r) {

            // IMPORTANT: Create a thread-local copy of the NDT state needed for derivatives
            // This prevents race conditions on j_ang_, h_ang_ etc.
            SvnNormalDistributionsTransform<PointSource, PointTarget> local_ndt = *this;

            Vector6d grad_k;
            Matrix6d hess_k;

            for (int k = r.begin(); k < r.end(); ++k) {
                // 1. Transform source cloud
                pcl::transformPointCloud(source_cloud, transformed_clouds[k], particles[k].matrix().cast<float>());

                // 2. Get NDT pose vector [x,y,z,r,p,y]
                Vector6d p_k_ndt;
                gtsam::Vector3 rpy = particles[k].rotation().rpy();
                p_k_ndt.head<3>() = particles[k].translation();
                p_k_ndt.tail<3>() = rpy;

                // 3. Compute derivatives using the thread-local copy
                double score_k = local_ndt.computeParticleDerivatives(grad_k, hess_k, transformed_clouds[k], p_k_ndt, true);

                // 4. Store negatives for loss minimization
                loss_gradients[k] = -grad_k;
                loss_hessians[k] = -hess_k; // Make sure hess_k is valid
                 if (!hess_k.allFinite()) {
                     PCL_WARN("[SvnNdt::align] NaN/Inf in NDT Hessian for particle %d, iter %d. Using Identity.\n", k, iter);
                     loss_hessians[k] = I6; // Use identity as fallback? Or zero? Identity might be safer.
                 }
            }
        }); // End TBB Stage 1

        // --- Stage 2: Calculate SVN Updates (Parallel using TBB) ---
        tbb::parallel_for(tbb::blocked_range<int>(0, K_),
            [&](const tbb::blocked_range<int>& r) {

            for (int k = r.begin(); k < r.end(); ++k) {
                Vector6d phi_k_star = Vector6d::Zero();
                Matrix6d H_k_tilde = Matrix6d::Zero();

                for (int l = 0; l < K_; ++l) {
                    double k_val = rbf_kernel(particles[l], particles[k]);
                    Vector6d k_grad = rbf_kernel_gradient(particles[l], particles[k]);

                    // Check for NaN/Inf in kernel/gradient
                     if (!std::isfinite(k_val) || !k_grad.allFinite()) {
                         PCL_WARN("[SvnNdt::align] NaN/Inf in kernel computation between %d and %d.\n", l, k);
                         continue; // Skip contribution from this pair
                     }

                    phi_k_star += k_val * loss_gradients[l] + k_grad;

                    // if (loss_hessians[l].allFinite()) { // <-- Restore check and use NDT Hessian
                    //     H_k_tilde += (k_val * k_val) * loss_hessians[l] + (k_grad * k_grad.transpose());
                    // } else {
                    //     H_k_tilde += (k_grad * k_grad.transpose()); // <-- Keep fallback
                    // }
                    H_k_tilde += (k_grad * k_grad.transpose());
                }

                phi_k_star /= static_cast<double>(K_);
                H_k_tilde /= static_cast<double>(K_);
                H_k_tilde += 1e-4 * I6; // Regularization

                Eigen::LDLT<Matrix6d> solver(H_k_tilde);
                if (solver.info() == Eigen::Success && H_k_tilde.allFinite()) {
                     Vector6d update = solver.solve(phi_k_star);
                     if (update.allFinite()) {
                         particle_updates[k] = update;
                     } else {
                         PCL_WARN("[SvnNdt::align] Solver produced NaN/Inf update for particle %d, iter %d. Setting update to zero.\n", k, iter);
                         particle_updates[k].setZero();
                     }
                } else {
                    PCL_WARN("[SvnNdt::align] LDLT solver failed or H_tilde invalid for particle %d, iter %d. Setting update to zero.\n", k, iter);
                    particle_updates[k].setZero();
                }
            }
        }); // End TBB Stage 2


        // --- Stage 3: Apply Updates (Serial) ---
        double total_update_norm = 0.0;
        for (int k = 0; k < K_; ++k) {
            Vector6d scaled_update = -step_size_ * particle_updates[k];
             if (!scaled_update.allFinite()) {
                 PCL_WARN("[SvnNdt::align] NaN/Inf in scaled update for particle %d, iter %d. Skipping update.\n", k, iter);
                 continue; // Skip update for this particle if invalid
             }
            total_update_norm += particle_updates[k].norm(); // Use unscaled norm for check
            particles[k] = particles[k].retract(scaled_update);
        }

        // --- Check Convergence ---
        result.iterations = iter + 1;
        double avg_update_norm = (K_ > 0) ? (total_update_norm / static_cast<double>(K_)) : 0.0;

        // ################'DEBUG
        std::cout << "[SVN Iter " << iter << "] Avg Update Norm: " << avg_update_norm << std::endl; 
        // ################'DEBUG

        if (avg_update_norm < stop_thresh_) {
            result.converged = true;
            break;
        }

    } // End SVN iteration loop

    // --- Finalization: Compute Mean Pose and Covariance ---
    if (K_ > 0) {
        Vector6d mean_xi = Vector6d::Zero();
        std::vector<Vector6d> tangent_vectors(K_);
        for(int k=0; k < K_; ++k) {
            tangent_vectors[k] = gtsam::Pose3::Logmap(prior_mean.between(particles[k]));
            mean_xi += tangent_vectors[k];
        }
        mean_xi /= static_cast<double>(K_);
        result.final_pose = prior_mean.retract(mean_xi);

        if (K_ > 1) {
            result.final_covariance.setZero();
            for(int k=0; k < K_; ++k) {
                Vector6d diff = tangent_vectors[k] - mean_xi;
                result.final_covariance += diff * diff.transpose();
            }
            result.final_covariance /= static_cast<double>(K_ - 1);
        } else {
            result.final_covariance = prior_noise_model->covariance(); // Use initial noise if K=1
        }

        // Final regularization (optional, but recommended)
        Eigen::SelfAdjointEigenSolver<Matrix6d> final_eigensolver(result.final_covariance);
        if (final_eigensolver.info() == Eigen::Success) {
            Vector6d final_evals = final_eigensolver.eigenvalues();
            if (final_evals(0) < 1e-7) { // Check smallest eigenvalue
                //PCL_DEBUG("[SvnNdt::align] Final covariance near singular. Adding regularization.\n");
                result.final_covariance += 1e-7 * I6;
            }
        } else {
             PCL_WARN("[SvnNdt::align] Eigendecomposition failed for final covariance. Matrix might be invalid.\n");
             // Maybe set covariance to identity * large value?
             result.final_covariance = 1e6 * I6;
        }

    } else { // Handle K_=0 case
        result.final_pose = prior_mean;
        result.final_covariance.setIdentity(); // Return identity if no particles
    }


    input_.reset(); // Clear internal pointer

    // Check if result converged based on iterations if not already set
    if (!result.converged && result.iterations == max_iter_) {
        // Did not converge within max iterations
        PCL_DEBUG("[SvnNdt::align] Reached max iterations (%d) without converging.\n", max_iter_);
    } else if (result.converged) {
         // Converged successfully
         // std::cout << "[SvnNdt::align] Converged in " << result.iterations << " iterations." << std::endl;
    }


    return result;
}


} // namespace svn_ndt

#endif // SVN_NDT_SVN_NDT_IMPL_HPP_