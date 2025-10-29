#ifndef SVN_NDT_SVN_NDT_H_
#define SVN_NDT_SVN_NDT_H_

// --- Standard/External Library Includes ---
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

// --- GTSAM Includes ---
#include <gtsam/geometry/Pose3.h> // For representing poses on SE(3) manifold

// --- PCL Includes ---
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// --- Custom Voxel Grid Include ---
#include <voxel_grid_covariance.h> // Include definition of VoxelGridCovariance and Leaf

namespace svn_ndt
{

// Forward declaration if needed elsewhere (though included above)
// template <typename PointSource, typename PointTarget>
// class SvnNormalDistributionsTransform;

/**
 * @brief Enum defining the available neighbor search methods for NDT voxel lookup.
 */
enum class NeighborSearchMethod
{
    KDTREE,  //!< Use K-D Tree radius search on valid voxel centroids. Requires VoxelGridCovariance::filter(true).
    DIRECT7, //!< Check the voxel containing the point and its 6 direct face neighbors.
    DIRECT1  //!< Check only the single voxel containing the point.
};

/**
 * @brief Structure to hold the results of the SVN-NDT alignment process.
 */
struct SvnNdtResult
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Good practice for structs potentially used with Eigen

    gtsam::Pose3 final_pose;                     //!< The final estimated mean pose on SE(3).
    Eigen::Matrix<double, 6, 6> final_covariance; //!< The final 6x6 covariance matrix in the tangent space at final_pose. GTSAM tangent space order: [roll, pitch, yaw, x, y, z].
    bool converged = false;                       //!< Flag indicating if the SVN iterations converged based on the stop threshold.
    int iterations = 0;                         //!< Number of SVN iterations performed.

    // Default constructor initializes pose to identity and covariance to zero
    SvnNdtResult() : final_pose(gtsam::Pose3()), final_covariance(Eigen::Matrix<double, 6, 6>::Zero()) {}
};


/**
 * @brief Implements Normal Distributions Transform (NDT) scan matching using Stein Variational Newton (SVN) for optimization and uncertainty estimation.
 *
 * This class aligns a source point cloud to a target NDT map (represented by VoxelGridCovariance).
 * It approximates the posterior pose distribution using a set of particles, optimizing their
 * positions using SVN, which leverages the gradient and Hessian of the NDT score function.
 * Designed with potential for TBB parallelism in the implementation (.hpp file).
 *
 * @tparam PointSource The PCL point type for the input (source) cloud.
 * @tparam PointTarget The PCL point type used to build the target NDT map.
 */
template <typename PointSource, typename PointTarget>
class SvnNormalDistributionsTransform
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Necessary if class holds fixed-size Eigen members directly (though not the case here)

    // --- Type Aliases ---
    using PointCloudSource = pcl::PointCloud<PointSource>;
    using PointCloudSourcePtr = typename PointCloudSource::Ptr;
    using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

    using PointCloudTarget = pcl::PointCloud<PointTarget>;
    using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
    using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

    // Eigen type aliases for clarity
    using Vector6d = Eigen::Matrix<double, 6, 1>;
    using Matrix6d = Eigen::Matrix<double, 6, 6>;

    // Alias for the LeafConstPtr type defined within VoxelGridCovariance
    using LeafConstPtr = typename svn_ndt::VoxelGridCovariance<PointTarget>::LeafConstPtr;

    // --- Constructor / Destructor ---
    /** @brief Default constructor, initializes parameters. */
    SvnNormalDistributionsTransform();
    /** @brief Virtual destructor. */
    virtual ~SvnNormalDistributionsTransform() = default; // Use default destructor

    // --- Core NDT Setup ---

    /**
     * @brief Provides the target point cloud used to build the internal NDT voxel map.
     * @details This triggers the creation/update of the internal VoxelGridCovariance object.
     * @param cloud Const pointer to the target point cloud.
     */
    void setInputTarget(const PointCloudTargetConstPtr& cloud);

    /**
     * @brief Sets the resolution (voxel size) for the target NDT map.
     * @details Rebuilds the internal voxel grid if a target cloud has already been set.
     * @param resolution The side length of the cubic voxels. Must be positive.
     */
    void setResolution(float resolution);

    /**
     * @brief Sets the minimum number of points required for a voxel to be valid.
     * @details This is passed to the internal VoxelGridCovariance. Must be >= 3.
     * @param min_points Minimum number of points.
     */
    void setMinPointPerVoxel(int min_points);

    /** @brief Gets the current resolution used for the NDT voxel grid. */
    float getResolution() const { return resolution_; }

    // --- Neighbor Search Method ---
    /** @brief Sets the method used to find neighboring voxels during NDT derivative calculation. */
    void setNeighborhoodSearchMethod(NeighborSearchMethod method) { search_method_ = method; }
    /** @brief Gets the currently selected neighbor search method. */
    NeighborSearchMethod getNeighborhoodSearchMethod() const { return search_method_; }

    // --- SVN Hyperparameters ---
    /** @brief Sets the number of particles used to approximate the posterior distribution. */
    void setParticleCount(int k) { K_ = (k > 0) ? k : 1; } // Ensure at least 1 particle
    /** @brief Gets the number of particles. */
    int getParticleCount() const { return K_; }

    /** @brief Sets the maximum number of SVN iterations. */
    void setMaxIterations(int iter) { max_iter_ = (iter > 0) ? iter : 1; } // Ensure at least 1 iteration
    /** @brief Gets the maximum number of SVN iterations. */
    int getMaxIterations() const { return max_iter_; }

    /** @brief Sets the bandwidth (h) for the RBF kernel used in SVN particle interactions. */
    void setKernelBandwidth(double h) { kernel_h_ = (h > 1e-9) ? h : 1e-9; } // Prevent zero/negative bandwidth
    /** @brief Gets the RBF kernel bandwidth. */
    double getKernelBandwidth() const { return kernel_h_; }

    /** @brief Sets the step size (epsilon) controlling particle movement per iteration. */
    void setStepSize(double eps) { step_size_ = (eps > 0) ? eps : 1e-6; } // Ensure positive step size
    /** @brief Gets the step size. */
    double getStepSize() const { return step_size_; }

    /** @brief Sets the threshold for early termination based on the average particle update norm. */
    void setEarlyStopThreshold(double thresh) { stop_thresh_ = (thresh >= 0) ? thresh : 1e-4; } // Ensure non-negative
    /** @brief Gets the early termination threshold. */
    double getEarlyStopThreshold() const { return stop_thresh_; }

    // --- NDT Specific Parameter ---
    /** @brief Sets the outlier ratio used in the NDT score calculation (influences d1, d2 constants). */
    void setOutlierRatio(double ratio); // Implementation updates NDT constants
    /** @brief Gets the outlier ratio. */
    double getOutlierRatio() const { return outlier_ratio_; }

    // --- Main Alignment Function ---
    /**
     * @brief Aligns the source point cloud to the target NDT map using SVN-NDT.
     * @param source_cloud The input source point cloud to align.
     * @param prior_mean An initial estimate (prior mean) for the transformation (source to target) as a gtsam::Pose3.
     * @return An SvnNdtResult struct containing the final estimated mean pose, its covariance,
     * and convergence information.
     */
    SvnNdtResult align(
        const PointCloudSource& source_cloud,
        const gtsam::Pose3& prior_mean
    ); // Implementation in .hpp file

// Make internal methods public temporarily for testing, if necessary
// #ifdef SVN_NDT_TESTING
public:
// #else
// protected: // Keep these protected for normal use
// #endif

    // --- Core NDT Math Functions (Adapted for SVN-NDT context) ---

    /**
     * @brief Computes the NDT score, its gradient, and Hessian for a single particle's pose.
     * @param[out] score_gradient Gradient of the NDT score w.r.t. the pose parameters.
     * @param[out] hessian Hessian of the NDT score w.r.t. the pose parameters.
     * @param[in] trans_cloud The source cloud transformed by the current particle's pose.
     * @param[in] p The particle's pose represented in the **[x, y, z, roll, pitch, yaw]** format required by the analytical derivative functions.
     * @param[in] compute_hessian Flag to enable/disable Hessian computation.
     * @return The NDT score (negative log-likelihood, lower is better).
     * @warning This function expects the pose 'p' in [x,y,z,r,p,y] order, requiring conversion from gtsam::Pose3.
     */
    double computeParticleDerivatives(
        Vector6d& score_gradient,
        Matrix6d& hessian,
        const PointCloudSource& trans_cloud,
        const Vector6d& p, // NDT [x,y,z,r,p,y] pose vector
        bool compute_hessian = true);

    /**
     * @brief Updates the gradient and Hessian based on a single point's contribution to the NDT score.
     * @details Called internally by computeParticleDerivatives.
     * @param[in,out] score_gradient Accumulated gradient.
     * @param[in,out] hessian Accumulated Hessian.
     * @param[in] point_gradient4 Precomputed Jacobian of the point transformation w.r.t pose p.
     * @param[in] point_hessian Precomputed Hessian of the point transformation w.r.t pose p.
     * @param[in] x_trans Transformed point relative to the voxel mean (point - mean).
     * @param[in] c_inv Inverse covariance matrix of the voxel.
     * @param[in] compute_hessian Flag to enable/disable Hessian update.
     * @param[in] use_gauss_newton_hessian Flag to use Gauss-Newton approximation for Hessian.
     * @return The score contribution of this point-voxel interaction.
     */
    double updateDerivatives(
        Vector6d& score_gradient,
        Matrix6d& hessian,
        const Eigen::Matrix<float, 4, 6>& point_gradient4, // Expects [x,y,z,r,p,y] convention
        const Eigen::Matrix<float, 24, 6>& point_hessian,  // Expects [x,y,z,r,p,y] convention
        const Eigen::Vector3d& x_trans,
        const Eigen::Matrix3d& c_inv,
        bool compute_hessian = true,
        bool use_gauss_newton_hessian = true) const; // <<< ADDED CONST


    /**
     * @brief Precomputes the angular portions of the Jacobian and Hessian matrices.
     * @param p The pose vector in **[x, y, z, roll, pitch, yaw]** format.
     * @param compute_hessian Flag to enable/disable Hessian precomputation.
     * @warning Expects pose 'p' in [x,y,z,r,p,y] order. Modifies member variables j_ang_ and h_ang_.
     */
    void computeAngleDerivatives(const Vector6d& p, bool compute_hessian = true); // Remains non-const

    /**
     * @brief Computes the Jacobian and Hessian of a single point's transformation w.r.t. the pose parameters.
     * @param x The original coordinates of the point (before transformation).
     * @param[out] point_gradient Jacobian matrix (point transformation w.r.t. pose p).
     * @param[out] point_hessian Hessian tensor (represented flattened) (point transformation w.r.t. pose p).
     * @param compute_hessian Flag to enable/disable Hessian computation.
     * @warning Assumes internal angular derivatives (`j_ang_`, `h_ang_`) have been precomputed based on the [x,y,z,r,p,y] pose.
     */
    void computePointDerivatives(
        const Eigen::Vector3d& x,                     // Original point
        Eigen::Matrix<float, 4, 6>& point_gradient, // Output Jacobian for [x,y,z,r,p,y]
        Eigen::Matrix<float, 24, 6>& point_hessian, // Output Hessian for [x,y,z,r,p,y]
        bool compute_hessian = true) const; // <<< ADDED CONST


    // --- SVN Helper Functions ---

    /**
     * @brief Computes the value of the Radial Basis Function (RBF) kernel between two poses.
     * @details Uses the squared norm of the difference in the SE(3) tangent space (via Logmap).
     * @param pose_l First pose (gtsam::Pose3).
     * @param pose_k Second pose (gtsam::Pose3).
     * @return Kernel value (scalar, typically between 0 and 1).
     */
    double rbf_kernel(const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const; // Already const

    /**
     * @brief Computes the gradient of the RBF kernel with respect to the *first* pose argument.
     * @details Uses the tangent space difference approximation.
     * @param pose_l First pose (gtsam::Pose3), the one the gradient is taken w.r.t.
     * @param pose_k Second pose (gtsam::Pose3).
     * @return Gradient vector (6x1) in the tangent space at pose_l.
     */
    Vector6d rbf_kernel_gradient(const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const; // Already const

    // --- Utility ---
    /** @brief Helper function to recalculate NDT constants (gauss_d1_, d2_, d3_) based on resolution and outlier_ratio_. */
    void updateNdtConstants();

    // --- Member Variables ---

    // NDT specific
    VoxelGridCovariance<PointTarget> target_cells_; //!< The internal NDT map representation.
    float resolution_ = 1.0f;                       //!< Voxel grid resolution.
    double outlier_ratio_ = 0.55;                   //!< Outlier ratio for NDT score calculation.
    double gauss_d1_{}, gauss_d2_{}, gauss_d3_{};   //!< NDT score calculation constants derived from resolution and outlier ratio.

    // Precomputed angle derivatives (State managed carefully within implementation)
    // Note: These are based on the [x,y,z,r,p,y] convention
    Eigen::Matrix<float, 8, 4> j_ang_;              //!< Precomputed angular Jacobian components.
    Eigen::Matrix<float, 16, 4> h_ang_;             //!< Precomputed angular Hessian components.

    // Search Method
    NeighborSearchMethod search_method_ = NeighborSearchMethod::DIRECT7; //!< Voxel neighbor search strategy.

    // SVN specific
    int K_ = 30;                 //!< Number of particles.
    int max_iter_ = 50;          //!< Maximum SVN iterations.
    double kernel_h_ = 0.1;      //!< RBF kernel bandwidth.
    double step_size_ = 0.1;     //!< SVN particle update step size scaling factor.
    double stop_thresh_ = 1e-4;  //!< Convergence threshold for average particle update norm.

    // Internal pointer to the source cloud during alignment (used by derivative functions)
    PointCloudSourceConstPtr input_; //!< Const pointer to the current source cloud being aligned.

}; // End class SvnNormalDistributionsTransform

} // namespace svn_ndt

// Include the implementation file (.hpp for templates)
#include <svn_ndt_impl.hpp>

#endif // SVN_NDT_SVN_NDT_H_