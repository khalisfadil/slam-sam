/*
 * TEST: test_convergence_comparison.cpp
 *
 * This test file is designed to compare the final convergence accuracy and performance
 * of the standard PCLOMP NDT against the SVN-NDT implementation.
 *
 * It removes all derivative-level comparisons and focuses on the final
 * alignment result from a known initial guess to a known ground truth.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <random> // For adding noise
#include <chrono> // For timing
#include <iomanip> // For std::fixed, std::setprecision

// --- Include your SVN-NDT header ---
#include "svn_ndt.h"
// --- Include PCLOMP NDT header ---
#include "pclomp/ndt_omp.h"

// --- PCL Includes ---
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h> // For pcl::transformPointCloud

// --- GTSAM Includes ---
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/base/Vector.h>

// Define point types for clarity
using PointT = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointT>;

/**
 * @brief Creates a synthetic source cloud and a noisy, transformed target cloud.
 * @param[in] ground_truth_pose The known transformation from source to target.
 * @param[in] noise_stddev Standard deviation of Gaussian noise to add to target points.
 * @param[out] source_cloud The generated source cloud (a clean 3D structure).
 * @param[out] target_cloud The generated target cloud (noisy and transformed).
 */
void create_test_clouds(const gtsam::Pose3& ground_truth_pose, double noise_stddev,
                        PointCloud& source_cloud, PointCloud& target_cloud)
{
    source_cloud.clear();
    target_cloud.clear();
    // Use a fixed seed for reproducibility
    std::default_random_engine gen(1337);
    std::normal_distribution<double> noise(0.0, noise_stddev);

    // --- Create Source Cloud ---
    // A structured cloud (two perpendicular planes)
    for (double x = -10.0; x <= 10.0; x += 0.5) {
        for (double y = -10.0; y <= 10.0; y += 0.5) {
            source_cloud.points.emplace_back(x, y, 0.0);
        }
    }
    for (double x = -10.0; x <= 10.0; x += 0.5) {
        for (double z = -10.0; z <= 10.0; z += 0.5) {
            source_cloud.points.emplace_back(x, 0.0, z);
        }
    }
    source_cloud.width = source_cloud.points.size();
    source_cloud.height = 1;
    source_cloud.is_dense = true;

    // --- Create Target Cloud ---
    target_cloud.points.reserve(source_cloud.points.size());
    for (const auto& src_pt : source_cloud.points) {
        gtsam::Point3 p_src(src_pt.x, src_pt.y, src_pt.z);
        gtsam::Point3 p_tgt = ground_truth_pose.transformFrom(p_src);
        PointT tgt_pt;
        tgt_pt.x = p_tgt.x() + noise(gen);
        tgt_pt.y = p_tgt.y() + noise(gen);
        tgt_pt.z = p_tgt.z() + noise(gen);
        target_cloud.points.push_back(tgt_pt);
    }
    target_cloud.width = target_cloud.points.size();
    target_cloud.height = 1;
    target_cloud.is_dense = true;
}


// --- Test Globals (for consistent comparison) ---
namespace {
    gtsam::Pose3 g_ground_truth_pose;
    gtsam::Pose3 g_initial_guess_pose;
    PointCloud::Ptr g_source_cloud;
    PointCloud::Ptr g_target_cloud;
    bool g_test_data_generated = false;

    // Tolerances for assertion
    const double g_trans_tolerance = 0.05; // 5cm
    const double g_rot_tolerance = 0.035;  // ~2 degrees

    void setup_global_test_data() {
        if (g_test_data_generated) return;

        std::cout << "--- Setting up global test data ---" << std::endl;

        // 1. Define Ground Truth
        gtsam::Rot3 R_gt = gtsam::Rot3::Yaw(0.2618) * gtsam::Rot3::Pitch(0.0873); // ~15 deg yaw, 5 deg pitch
        gtsam::Point3 t_gt(0.5, 0.0, 0.3); // 50cm x, 30cm z
        g_ground_truth_pose = gtsam::Pose3(R_gt, t_gt);

        // 2. Define Initial Guess (with a small error)
        // Small rotation and translation error
        gtsam::Vector6 delta_xi; delta_xi << 0.05, -0.02, 0.04, 0.02, -0.01, 0.03;
        g_initial_guess_pose = g_ground_truth_pose.retract(-delta_xi); // Retract *negative* delta to get guess near gt

        std::cout << "Ground Truth Pose:\n" << g_ground_truth_pose << std::endl;
        std::cout << "Initial Guess Pose:\n" << g_initial_guess_pose << std::endl;

        // 3. Generate Clouds
        // *** FIX: Use .reset(new ...) to be compatible with both
        // std::shared_ptr and boost::shared_ptr (which Ptr might alias) ***
        g_source_cloud.reset(new PointCloud());
        g_target_cloud.reset(new PointCloud());

        double noise_stddev = 0.02; // 2cm noise
        create_test_clouds(g_ground_truth_pose, noise_stddev, *g_source_cloud, *g_target_cloud);

        ASSERT_FALSE(g_source_cloud->empty()) << "Global source cloud generation failed.";
        ASSERT_FALSE(g_target_cloud->empty()) << "Global target cloud generation failed.";
        std::cout << "Generated source (" << g_source_cloud->size() << " pts) and target (" << g_target_cloud->size() << " pts)" << std::endl;
        std::cout << "--------------------------------------" << std::endl;

        g_test_data_generated = true;
    }
} // anonymous namespace


/**
 * @brief Test case for the standard PCLOMP NDT implementation.
 */
TEST(ConvergenceComparison, PclOmp) {
    // 1. --- Get Global Test Data ---
    setup_global_test_data();
    ASSERT_TRUE(g_test_data_generated);

    // 2. --- Configure and Run PCLOMP NDT ---
    pclomp::NormalDistributionsTransform<PointT, PointT> ndt;
    
    // Set common parameters
    ndt.setResolution(1.0f);
    ndt.setNeighborhoodSearchMethod(pclomp::NeighborSearchMethod::DIRECT7);
    ndt.setMaximumIterations(50);
    ndt.setTransformationEpsilon(1e-4);
    ndt.setStepSize(0.1); // Standard NDT step size
    ndt.setNumThreads(10); // Use 4 threads for comparison
    
    std::cout << "[PCLOMP Test] Using " << ndt.getNumThreads() << " OpenMP threads." << std::endl;

    // ---
    // *** FIX: SET BOTH THE TARGET AND THE SOURCE CLOUDS ***
    // ---
    ndt.setInputTarget(g_target_cloud);
    ndt.setInputSource(g_source_cloud); // <-- THIS LINE WAS MISSING
    // ---

    // Convert gtsam::Pose3 to Eigen::Matrix4f for PCLOMP
    Eigen::Matrix4f initial_guess_matrix = g_initial_guess_pose.matrix().cast<float>();
    PointCloud final_cloud; // Output cloud

    std::cout << "[PCLOMP Test] Starting PCLOMP alignment..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run alignment
    ndt.computeTransformation(final_cloud, initial_guess_matrix);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> align_time_ms = end_time - start_time;

    // 3. --- Check Results ---
    bool converged = ndt.hasConverged();
    int iterations = ndt.getFinalNumIteration();
    Eigen::Matrix4f final_transform_matrix = ndt.getFinalTransformation();
    gtsam::Pose3 final_pose(final_transform_matrix.cast<double>()); // Convert back to gtsam::Pose3

    std::cout << "[PCLOMP Test] Alignment finished in " << align_time_ms.count() << " ms." << std::endl;
    std::cout << "[PCLOMP Test] Converged: " << (converged ? "True" : "False") << " in " << iterations << " iterations." << std::endl;

    EXPECT_TRUE(converged) << "PCLOMP Test failed to converge!";
    EXPECT_LT(iterations, 50) << "PCLOMP Test took maximum iterations (" << iterations << ")";

    gtsam::Pose3 error_pose = final_pose.between(g_ground_truth_pose);
    gtsam::Vector6 error_log = gtsam::Pose3::Logmap(error_pose);
    double trans_error_norm = error_log.tail(3).norm();
    double rot_error_norm = error_log.head(3).norm();

    std::cout << "[PCLOMP Test] Estimated Pose: \n" << final_pose << std::endl;
    std::cout << "[PCLOMP Test] Translation Error Norm: " << trans_error_norm << " m" << std::endl;
    std::cout << "[PCLOMP Test] Rotation Error Norm:    " << rot_error_norm << " rad" << std::endl;

    ASSERT_LT(trans_error_norm, g_trans_tolerance) << "PCLOMP Test Translation error is too high.";
    ASSERT_LT(rot_error_norm, g_rot_tolerance) << "PCLOMP Test Rotation error is too high.";
}


/**
 * @brief Test case for the SVN-NDT implementation with K=10 particles.
 */
TEST(ConvergenceComparison, SvnNdtK10) {
    // 1. --- Get Global Test Data ---
    setup_global_test_data();
    ASSERT_TRUE(g_test_data_generated);

    // 2. --- Configure and Run SVN-NDT with K=10 ---
    svn_ndt::SvnNormalDistributionsTransform<PointT, PointT> ndt;
    
    // Set common parameters
    ndt.setResolution(1.0f);
    ndt.setMinPointPerVoxel(3);
    ndt.setNeighborhoodSearchMethod(svn_ndt::NeighborSearchMethod::DIRECT7);
    ndt.setNumThreads(10); // Use 4 threads for comparison
    // ndt.setUseGaussNewtonHessian(true); // (true is default, good for SVN)

    // Set SVN parameters
    ndt.setParticleCount(10);      // K=10 particles
    ndt.setMaxIterations(50);
    ndt.setKernelBandwidth(1.0);   
    ndt.setEarlyStopThreshold(1e-4);
    ndt.setStepSize(0.5); // <-- TUNING FIX: Reduced step size from 1.5 to 0.5
    
    std::cout << "[SVN-NDT K=10 Test] Using " << ndt.getNumThreads() << " OpenMP threads." << std::endl;

    ndt.setInputTarget(g_target_cloud);

    std::cout << "[SVN-NDT K=10 Test] Starting SVN-NDT alignment (K=10)..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    svn_ndt::SvnNdtResult result = ndt.align(*g_source_cloud, g_initial_guess_pose);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> align_time_ms = end_time - start_time;

    // 3. --- Check Results ---
    std::cout << "[SVN-NDT K=10 Test] Alignment finished in " << align_time_ms.count() << " ms." << std::endl;
    std::cout << "[SVN-NDT K=10 Test] Converged: " << (result.converged ? "True" : "False") << " in " << result.iterations << " iterations." << std::endl;

    EXPECT_TRUE(result.converged) << "K=10 SVN-NDT Test failed to converge!";
    EXPECT_LT(result.iterations, 50) << "K=10 SVN-NDT Test took maximum iterations (" << result.iterations << ")";

    gtsam::Pose3 error_pose = result.final_pose.between(g_ground_truth_pose);
    gtsam::Vector6 error_log = gtsam::Pose3::Logmap(error_pose);
    double trans_error_norm = error_log.tail(3).norm();
    double rot_error_norm = error_log.head(3).norm();

    std::cout << "[SVN-NDT K=10 Test] Estimated Pose: \n" << result.final_pose << std::endl;
    std::cout << "[SVN-NDT K=10 Test] Translation Error Norm: " << trans_error_norm << " m" << std::endl;
    std::cout << "[SVN-NDT K=10 Test] Rotation Error Norm:    " << rot_error_norm << " rad" << std::endl;

    ASSERT_LT(trans_error_norm, g_trans_tolerance) << "K=10 SVN-NDT Test Translation error is too high.";
    ASSERT_LT(rot_error_norm, g_rot_tolerance) << "K=10 SVN-NDT Test Rotation error is too high.";
}


/**
 * @brief Test case for the SVN-NDT implementation with K=1 particle (Pure Newton's Method).
 */
TEST(ConvergenceComparison, SvnNdtK1_Newton) {
    // 1. --- Get Global Test Data ---
    setup_global_test_data();
    ASSERT_TRUE(g_test_data_generated);

    // 2. --- Configure and Run SVN-NDT with K=1 ---
    svn_ndt::SvnNormalDistributionsTransform<PointT, PointT> ndt;

    // Set common parameters
    ndt.setResolution(1.0f);
    ndt.setMinPointPerVoxel(3);
    ndt.setNeighborhoodSearchMethod(svn_ndt::NeighborSearchMethod::DIRECT7);
    ndt.setNumThreads(10); // Use 4 threads for comparison

    // Set SVN parameters for K=1 (Newton's method)
    ndt.setParticleCount(1);       // K=1 particle
    ndt.setMaxIterations(50);
    ndt.setKernelBandwidth(1.0);   // Not used for K=1
    ndt.setEarlyStopThreshold(1e-4);
    
    // --- TUNING FIX: Use full Newton step (1.0) and the full analytical Hessian ---
    ndt.setStepSize(0.5);
    ndt.setUseGaussNewtonHessian(false); // <-- USE FULL HESSIAN
    
    std::cout << "[SVN-NDT K=1 Test] Using " << ndt.getNumThreads() << " OpenMP threads." << std::endl;

    ndt.setInputTarget(g_target_cloud);

    std::cout << "[SVN-NDT K=1 Test] Starting SVN-NDT alignment (K=1, Full Newton Test)..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    svn_ndt::SvnNdtResult result = ndt.align(*g_source_cloud, g_initial_guess_pose);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> align_time_ms = end_time - start_time;

    // 3. --- Check Results ---
    std::cout << "[SVN-NDT K=1 Test] Alignment finished in " << align_time_ms.count() << " ms." << std::endl;
    std::cout << "[SVN-NDT K=1 Test] Converged: " << (result.converged ? "True" : "False") << " in " << result.iterations << " iterations." << std::endl;

    EXPECT_TRUE(result.converged) << "K=1 SVN-NDT Test failed to converge!";
    EXPECT_LT(result.iterations, 50) << "K=1 SVN-NDT Test took maximum iterations (" << result.iterations << ")";

    gtsam::Pose3 error_pose = result.final_pose.between(g_ground_truth_pose);
    gtsam::Vector6 error_log = gtsam::Pose3::Logmap(error_pose);
    double trans_error_norm = error_log.tail(3).norm();
    double rot_error_norm = error_log.head(3).norm();

    std::cout << "[SVN-NDT K=1 Test] Estimated Pose: \n" << result.final_pose << std::endl;
    std::cout << "[SVN-NDT K=1 Test] Translation Error Norm: " << trans_error_norm << " m" << std::endl;
    std::cout << "[SVN-NDT K=1 Test] Rotation Error Norm:    " << rot_error_norm << " rad" << std::endl;

    ASSERT_LT(trans_error_norm, g_trans_tolerance) << "K=1 SVN-NDT Test Translation error is too high.";
    ASSERT_LT(rot_error_norm, g_rot_tolerance) << "K=1 SVN-NDT Test Rotation error is too high.";
}