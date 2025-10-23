#ifndef SVN_NDT_SVN_NDT_H_
#define SVN_NDT_SVN_NDT_H_

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam/geometry/Pose3.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <voxel_grid_covariance.h> // Our voxel grid header

namespace svn_ndt
{

// Forward declaration
template <typename PointSource, typename PointTarget>
class SvnNormalDistributionsTransform;

/**
 * @brief Enum defining the available neighbor search methods for NDT lookup.
 */
enum class NeighborSearchMethod
{
    KDTREE,  //!< Use K-D Tree radius search on voxel centroids. Requires VoxelGridCovariance::filter(true).
    DIRECT7, //!< Check the current voxel and its 6 face neighbors.
    DIRECT1  //!< Check only the voxel containing the point.
};

/**
 * @brief Structure to hold the results of the SVN-NDT alignment.
 */
struct SvnNdtResult
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    gtsam::Pose3 final_pose;                     //!< The final estimated mean pose.
    Eigen::Matrix<double, 6, 6> final_covariance; //!< The final 6x6 sample covariance matrix (GTSAM tangent: r,p,y,x,y,z).
    bool converged = false;                       //!< Did the SVN iterations converge based on the threshold?
    int iterations = 0;                         //!< Number of SVN iterations performed.

    // Default constructor
    SvnNdtResult() : final_pose(gtsam::Pose3()), final_covariance(Eigen::Matrix<double, 6, 6>::Zero()) {}
};


/**
 * @brief Implements Normal Distributions Transform (NDT) scan matching
 * with uncertainty estimation using Stein Variational Newton (SVN).
 *
 * Approximates the posterior pose distribution using particles optimized via SVN,
 * leveraging NDT's gradient and Hessian. Designed for TBB parallelism.
 */
template <typename PointSource, typename PointTarget>
class SvnNormalDistributionsTransform
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // --- Type Aliases ---
    using PointCloudSource = pcl::PointCloud<PointSource>;
    using PointCloudSourcePtr = typename PointCloudSource::Ptr;
    using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

    using PointCloudTarget = pcl::PointCloud<PointTarget>;
    using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
    using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

    using Vector6d = Eigen::Matrix<double, 6, 1>;
    using Matrix6d = Eigen::Matrix<double, 6, 6>;
    // Alias for the LeafConstPtr type defined in VoxelGridCovariance
    using LeafConstPtr = typename svn_ndt::VoxelGridCovariance<PointTarget>::LeafConstPtr;

    // --- Constructor / Destructor ---
    SvnNormalDistributionsTransform();
    virtual ~SvnNormalDistributionsTransform() {}

    // --- Core NDT Setup ---
    void setInputTarget(const PointCloudTargetConstPtr& cloud);
    void setResolution(float resolution);
    float getResolution() const { return resolution_; }

    // --- Neighbor Search Method ---
    /** @brief Sets the method used to find neighboring voxels during derivative calculation. */
    void setNeighborhoodSearchMethod(NeighborSearchMethod method) { search_method_ = method; }
    /** @brief Gets the currently selected neighbor search method. */
    NeighborSearchMethod getNeighborhoodSearchMethod() const { return search_method_; }

    // --- SVN Hyperparameters ---
    void setParticleCount(int k) { K_ = k; }
    int getParticleCount() const { return K_; } // particles used to represent the approximation of the pose posterior distribution

    void setMaxIterations(int iter) { max_iter_ = iter; }
    int getMaxIterations() const { return max_iter_; } // maximum number of times the algorithm will update the positions of all particles

    void setKernelBandwidth(double h) { kernel_h_ = h; }
    double getKernelBandwidth() const { return kernel_h_; } // Controls the "radius of influence" between particles.
    //  Small h: Particles are strongly repelled by nearby particles but barely influenced by distant ones. This can lead to spiky, separated modes but might struggle to represent broad distributions smoothly.
    //  Large h: Particles influence each other over longer distances. This promotes smoother distributions but might cause particles to collapse towards a single mode too quickly, potentially missing multi-modality.

    void setStepSize(double eps) { step_size_ = eps; }
    double getStepSize() const { return step_size_; }   // Controls how far the particles move in each iteration.
    // Too large: Particles might overshoot the target distribution or become unstable.
    // Too small: Convergence will be very slow.

    void setEarlyStopThreshold(double thresh) { stop_thresh_ = thresh; } // average magnitude (norm) of the particle update vectors
    double getEarlyStopThreshold() const { return stop_thresh_; }       // Stops the iterative process early if the particles are barely moving anymore

    // --- NDT Specific Parameter ---
    void setOutlierRatio(double ratio); // Implementation updates gauss constants
    double getOutlierRatio() const { return outlier_ratio_; }
    // influences the shape of the negative log-likelihood function assigned to points based on their Mahalanobis distance from a voxel's mean.

    // --- Main Alignment Function ---
    /**
     * @brief Aligns the source cloud to the target using SVN-NDT.
     * @param source_cloud The input source point cloud.
     * @param prior_mean An initial guess for the transformation (mean of the prior distribution).
     * @return An SvnNdtResult struct containing the mean pose, covariance, and convergence status.
     */
    SvnNdtResult align(
        const PointCloudSource& source_cloud,
        const gtsam::Pose3& prior_mean
    );


protected: // --- Internal Methods and Data ---

    // --- Core NDT Math Functions (Implementations in svn_ndt_impl.hpp) ---
    double computeParticleDerivatives(
        Vector6d& score_gradient,
        Matrix6d& hessian,
        const PointCloudSource& trans_cloud,
        const Vector6d& p, // NDT [x,y,z,r,p,y] pose vector
        bool compute_hessian = true);

    double updateDerivatives(
        Vector6d& score_gradient,
        Matrix6d& hessian,
        const Eigen::Matrix<float, 4, 6>& point_gradient4,
        const Eigen::Matrix<float, 24, 6>& point_hessian_,
        const Eigen::Vector3d& x_trans,
        const Eigen::Matrix3d& c_inv,
        bool compute_hessian = true);

    void computeAngleDerivatives(const Vector6d& p, bool compute_hessian = true);

    void computePointDerivatives(
        const Eigen::Vector3d& x,
        Eigen::Matrix<float, 4, 6>& point_gradient_,
        Eigen::Matrix<float, 24, 6>& point_hessian_,
        bool compute_hessian = true);


    // --- SVN Helper Functions ---
    double rbf_kernel(const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const;
    Vector6d rbf_kernel_gradient(const gtsam::Pose3& pose_l, const gtsam::Pose3& pose_k) const;

    // --- Utility ---
    void updateNdtConstants(); // Helper to recalculate gauss_d1_, d2_, d3_

    // --- Member Variables ---

    // NDT specific
    VoxelGridCovariance<PointTarget> target_cells_;
    float resolution_ = 1.0f;
    double outlier_ratio_ = 0.55;
    double gauss_d1_{}, gauss_d2_{}, gauss_d3_{}; // NDT normalization constants

    // Precomputed angle derivatives (State managed carefully within computeParticleDerivatives)
    Eigen::Matrix<float, 8, 4> j_ang_;
    Eigen::Matrix<float, 16, 4> h_ang_;

    // Search Method
    NeighborSearchMethod search_method_ = NeighborSearchMethod::DIRECT7; // Default

    // SVN specific
    int K_ = 30;
    int max_iter_ = 50;
    double kernel_h_ = 0.1;
    double step_size_ = 0.1;
    double stop_thresh_ = 1e-4;

    // Internal pointer to the source cloud during alignment
    PointCloudSourceConstPtr input_; // Use ConstPtr

};

} // namespace svn_ndt

// Include the implementation file (.hpp for templates)
#include <svn_ndt_impl.hpp>

#endif // SVN_NDT_SVN_NDT_H_