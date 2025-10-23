/*
 * Software License Agreement (BSD License)
 *
 * Point Cloud Library (PCL) - www.pointclouds.org
 * Copyright (c) 2010-2011, Willow Garage, Inc.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 * * Neither the name of the copyright holder(s) nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Modifications for SVN-NDT: Changed namespace to svn_ndt, switched to tsl::robin_map.
 */

#ifndef SVN_NDT_VOXEL_GRID_COVARIANCE_H_ // Updated include guard
#define SVN_NDT_VOXEL_GRID_COVARIANCE_H_

// --- PCL Includes ---
// pcl_macros.h must come first
// clang-format off
#include <pcl/pcl_macros.h>
// clang-format on
#include <pcl/filters/boost.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/common/eigen.h> // For PCL Eigen types if needed

// --- Standard Library Includes ---
#include <vector>
#include <map> // Keep for potential compatibility if needed elsewhere

// --- External Library Includes ---
#include <robin_map.h> // Include the robin_map header
#include <Eigen/Core>      // For Eigen::Vector3d, Matrix3d etc.
#include <Eigen/Eigenvalues> // For covariance computation
#include <Eigen/StdVector>

namespace svn_ndt // Updated namespace
{
/**
 * @brief A searchable voxel structure containing the mean and covariance of point data.
 * @details This class inherits from PCL's VoxelGrid but overrides the filter application
 * to compute and store the covariance matrix and its inverse for each
 * voxel (leaf) containing enough points. It uses tsl::robin_map for
 * efficient storage and retrieval of voxel data. The core logic is based
 * on the NDT method described by Magnusson (2009).
 * @tparam PointT The PCL point type (e.g., pcl::PointXYZ, pcl::PointXYZI).
 */
template <typename PointT>
class VoxelGridCovariance : public pcl::VoxelGrid<PointT>
{
protected:
    // Make parent class's protected members available
    using pcl::VoxelGrid<PointT>::filter_name_;
    using pcl::VoxelGrid<PointT>::getClassName;
    using pcl::VoxelGrid<PointT>::input_;
    using pcl::VoxelGrid<PointT>::indices_;
    using pcl::VoxelGrid<PointT>::filter_limit_negative_;
    using pcl::VoxelGrid<PointT>::filter_limit_min_;
    using pcl::VoxelGrid<PointT>::filter_limit_max_;
    using pcl::VoxelGrid<PointT>::filter_field_name_;

    using pcl::VoxelGrid<PointT>::downsample_all_data_;
    using pcl::VoxelGrid<PointT>::leaf_layout_;
    using pcl::VoxelGrid<PointT>::save_leaf_layout_;
    using pcl::VoxelGrid<PointT>::leaf_size_;
    using pcl::VoxelGrid<PointT>::min_b_;
    using pcl::VoxelGrid<PointT>::max_b_;
    using pcl::VoxelGrid<PointT>::inverse_leaf_size_;
    using pcl::VoxelGrid<PointT>::div_b_;
    using pcl::VoxelGrid<PointT>::divb_mul_;

    // Typedefs for point cloud types
    using PointCloud = typename pcl::Filter<PointT>::PointCloud;
    using PointCloudPtr = typename PointCloud::Ptr;
    using PointCloudConstPtr = typename PointCloud::ConstPtr;

public:
    // Correct shared_ptr typedefs using the current class name
#if PCL_VERSION >= PCL_VERSION_CALC(1, 10, 0)
    using Ptr = pcl::shared_ptr<VoxelGridCovariance<PointT>>;
    using ConstPtr = pcl::shared_ptr<const VoxelGridCovariance<PointT>>;
#else
    using Ptr = boost::shared_ptr<VoxelGridCovariance<PointT>>;
    using ConstPtr = boost::shared_ptr<const VoxelGridCovariance<PointT>>;
#endif

    /**
     * @brief Structure holding statistical data for a single voxel (leaf).
     * @details Contains the number of points, 3D mean, 3D covariance matrix,
     * inverse covariance, and eigenvectors/eigenvalues. Optionally
     * stores an Nd centroid if `downsample_all_data_` is true.
     */
    struct alignas(32) Leaf
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Crucial for fixed-size Eigen members

        /** @brief Default constructor initializes numerical members to zero/identity. */
        Leaf()
            : nr_points_(0),
              mean_(Eigen::Vector3d::Zero()),
              centroid_(), // Will be resized if downsample_all_data_ is true
              cov_(Eigen::Matrix3d::Identity()),
              icov_(Eigen::Matrix3d::Zero()),
              evecs_(Eigen::Matrix3d::Identity()),
              evals_(Eigen::Vector3d::Zero())
        {
        }

        // --- Getters ---
        int getPointCount() const { return nr_points_; }
        const Eigen::Vector3d& getMean() const { return mean_; }
        const Eigen::Matrix3d& getCov() const { return cov_; }
        const Eigen::Matrix3d& getInverseCov() const { return icov_; }
        const Eigen::Matrix3d& getEvecs() const { return evecs_; }
        const Eigen::Vector3d& getEvals() const { return evals_; }

        // --- Public Members (Consider making private with getters if stricter encapsulation is desired) ---
        int nr_points_;             // Number of points falling within this voxel
        Eigen::Vector3d mean_;      // 3D Centroid (mean) of points in the voxel
        Eigen::VectorXf centroid_;  // Nd Centroid (if downsample_all_data_ is true)
        Eigen::Matrix3d cov_;       // 3x3 Covariance matrix of points
        Eigen::Matrix3d icov_;      // Inverse of the 3x3 covariance matrix
        Eigen::Matrix3d evecs_;     // Eigenvectors of the covariance matrix (columns)
        Eigen::Vector3d evals_;     // Eigenvalues corresponding to the eigenvectors
    };

    /** @brief Pointer to a Leaf structure. */
    using LeafPtr = Leaf*;
    /** @brief Const pointer to a Leaf structure. */
    using LeafConstPtr = const Leaf*;

    /** @brief Hash map storing voxel leaves, keyed by their computed 1D index. */
    // Define the pair type that will be stored
    using MapPair = std::pair<const size_t, Leaf>;

    // Tell robin_map to use Eigen's aligned allocator for this pair
    using Map = tsl::robin_map<size_t, Leaf, std::hash<size_t>, std::equal_to<size_t>,
                            Eigen::aligned_allocator<MapPair>>;

public:
    /**
     * @brief Default constructor. Initializes parameters.
     */
    VoxelGridCovariance()
        : searchable_(false),
          min_points_per_voxel_(6), // Default from NDT paper/implementations
          min_covar_eigvalue_mult_(0.01), // Regularization factor
          leaves_(),
          voxel_centroids_(),
          voxel_centroids_leaf_indices_(),
          kdtree_()
    {
        // Settings typically used for NDT
        this->downsample_all_data_ = false; // Only compute 3D cov usually
        this->save_leaf_layout_ = false; // Not needed unless debugging layout
        this->leaf_size_.setZero();
        this->min_b_.setZero();
        this->max_b_.setZero();
        this->filter_name_ = "SvnNdtVoxelGridCovariance"; // Specific name
    }

    // --- Configuration Methods ---

    /** @brief Set the minimum number of points required for a voxel's covariance to be computed and used. Must be >= 3. */
    inline void setMinPointPerVoxel(int min_points_per_voxel)
    {
        if (min_points_per_voxel >= 3)
        {
            min_points_per_voxel_ = min_points_per_voxel;
        }
        else
        {
            PCL_WARN("[%s::setMinPointPerVoxel] Covariance calculation requires at least 3 points. Setting to 3.\n", getClassName().c_str());
            min_points_per_voxel_ = 3;
        }
    }

    /** @brief Get the minimum number of points required per voxel. */
    inline int getMinPointPerVoxel() const { return min_points_per_voxel_; }

    /** @brief Set the regularization factor for covariance eigenvalues. Prevents singularity. */
    inline void setCovEigValueInflationRatio(double min_covar_eigvalue_mult)
    {
        min_covar_eigvalue_mult_ = min_covar_eigvalue_mult;
    }

    /** @brief Get the eigenvalue inflation ratio. */
    inline double getCovEigValueInflationRatio() const { return min_covar_eigvalue_mult_; }

    // --- Filtering and Grid Building ---

    /**
     * @brief Process the input cloud, compute voxel means/covariances, and optionally build a search structure.
     * @param[out] output Optional output cloud containing centroids of valid voxels.
     * @param[in] searchable If true, builds a k-d tree on valid voxel centroids for neighbor searches.
     */
    inline void filter(PointCloud& output, bool searchable = false)
    {
        searchable_ = searchable;
        applyFilter(output); // Calls the overridden method below

        // Need to build centroid cloud *after* applyFilter for kd-tree
        buildCentroidCloudAndIndexMap(); // Helper function

        if (searchable_ && !voxel_centroids_->empty())
        {
            kdtree_.setInputCloud(voxel_centroids_);
        }
         else if (searchable_) // Handle case where grid built but no valid centroids
        {
            kdtree_.setInputCloud(voxel_centroids_); // Set empty cloud
        }
    }

    /**
     * @brief Process the input cloud, compute voxel means/covariances, and optionally build a search structure (no output cloud).
     * @param[in] searchable If true, builds a k-d tree on valid voxel centroids.
     */
    inline void filter(bool searchable = false)
    {
        searchable_ = searchable;
        PointCloud dummy_output; // Temporary storage for base class if needed
        applyFilter(dummy_output); // Calls the overridden method below

        // Build centroid cloud and index mapping AFTER applyFilter populates leaves_
        buildCentroidCloudAndIndexMap(); // Helper function

        if (searchable_ && !voxel_centroids_->empty())
        {
            kdtree_.setInputCloud(voxel_centroids_);
        }
        else if (searchable_) // Handle case where grid built but no valid centroids
        {
            kdtree_.setInputCloud(voxel_centroids_); // Set empty cloud
        }
    }

    // --- Leaf Access and Search Methods ---

    /**
     * @brief Get a const pointer to the leaf corresponding to a specific voxel index.
     * @param index The 1D index computed for the voxel.
     * @return Const pointer to the Leaf structure, or nullptr if index not found or leaf invalid.
     */
    inline LeafConstPtr getLeaf(size_t index) const
    {
        auto it = leaves_.find(index);
        // Check nr_points >= min_points_per_voxel_ *after* finding the leaf
        if (it != leaves_.end() && it->second.nr_points_ >= min_points_per_voxel_) {
            return &(it->second);
        }
        return nullptr;
    }

    /**
     * @brief Get a const pointer to the leaf containing the given 3D point coordinates.
     * @param p Point coordinates.
     * @return Const pointer to the Leaf structure, or nullptr if voxel is empty or invalid or point outside bounds.
     */
    inline LeafConstPtr getLeaf(const Eigen::Vector3f& p) const
    {
        // Check grid bounds first (requires min_b_ and max_b_ to be computed by applyFilter)
        // Need to compare against float coords, not integer indices
        if (inverse_leaf_size_[0] == 0.0f) { // Check if grid is initialized
             PCL_WARN("[%s::getLeaf] Voxel grid not initialized (leaf size is zero).\n", getClassName().c_str());
             return nullptr;
        }
         Eigen::Vector4f pt4(p[0], p[1], p[2], 0.0f);
         if (!isPointWithinBounds(pt4)) {
             return nullptr;
         }

        // Compute 1D index using the base class computed min_b_ and multipliers
        int ijk0 = static_cast<int>(std::floor(p[0] * inverse_leaf_size_[0]) - min_b_[0]);
        int ijk1 = static_cast<int>(std::floor(p[1] * inverse_leaf_size_[1]) - min_b_[1]);
        int ijk2 = static_cast<int>(std::floor(p[2] * inverse_leaf_size_[2]) - min_b_[2]);
        size_t idx = static_cast<size_t>(ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2]);

        return getLeaf(idx);
    }

     /**
     * @brief Get a const pointer to the leaf containing the given PCL point.
     * @param p PCL point.
     * @return Const pointer to the Leaf structure, or nullptr if voxel is empty or invalid.
     */
    inline LeafConstPtr getLeaf(const PointT& p) const
    {
        // Delegate to the Eigen::Vector3f version
        return getLeaf(p.getVector3fMap()); // Use PCL's map function
    }


    /** @brief Get a const reference to the internal map holding all leaves (including invalid ones). */
    inline const Map& getAllLeaves() const { return leaves_; }

    /** @brief Get a const shared pointer to the point cloud of valid voxel centroids. Requires searchability enabled. */
    inline PointCloudConstPtr getCentroids() const { return voxel_centroids_; }

    /**
     * @brief Find K nearest valid voxel centroids to a query point using the k-d tree.
     * @details Requires `filter(true)` or `filter(output, true)` to have been called.
     * @param[in] point The query point.
     * @param[in] k The number of neighbors to search for.
     * @param[out] k_leaves Vector filled with const pointers to the found valid leaves.
     * @param[out] k_sqr_distances Vector filled with squared distances to the centroids.
     * @return Number of neighbors found (<= k).
     */
    int nearestKSearch( const PointT& point, int k,
                        std::vector<LeafConstPtr>& k_leaves,
                        std::vector<float>& k_sqr_distances) const
    {
        k_leaves.clear();
        k_sqr_distances.clear();

        if (!searchable_) {
            PCL_WARN("[%s::nearestKSearch] Grid not searchable. Call filter(true) first.\n", getClassName().c_str());
            return 0;
        }
        if (!kdtree_.getInputCloud() || voxel_centroids_->empty()) {
             // PCL_DEBUG("[%s::nearestKSearch] KdTree not initialized or no valid centroids.\n", getClassName().c_str());
             return 0; // No centroids to search
        }
        if (k <= 0) return 0; // Trivial case


        std::vector<int> k_indices; // Indices relative to voxel_centroids_
        int found_k = kdtree_.nearestKSearch(point, k, k_indices, k_sqr_distances);

        k_leaves.reserve(found_k);
        // k_sqr_distances is filled by kdtree_

        for (int centroid_idx : k_indices) {
            // Map kdtree index (index within voxel_centroids_) back to the original leaf index
             if (centroid_idx < 0 || static_cast<size_t>(centroid_idx) >= voxel_centroids_leaf_indices_.size()) {
                 PCL_ERROR("[%s::nearestKSearch] Invalid index %d from KdTree.\n", getClassName().c_str(), centroid_idx);
                 continue;
             }
            size_t leaf_map_idx = voxel_centroids_leaf_indices_[centroid_idx];
            LeafConstPtr leaf_ptr = getLeaf(leaf_map_idx); // Use safe getter which checks min points
            if (leaf_ptr) {
                k_leaves.push_back(leaf_ptr);
            } else {
                 // This might happen if getLeaf's min_points check fails, but the centroid was initially created.
                 // Or if index mapping is somehow incorrect.
                 // PCL_DEBUG("[%s::nearestKSearch] Consistency warning: Centroid index %d maps to leaf index %zu, but getLeaf returned null.\n",
                 //           getClassName().c_str(), centroid_idx, leaf_map_idx);
            }
        }
        // Ensure distance vector matches the number of leaves actually added
        k_sqr_distances.resize(k_leaves.size());
        return k_leaves.size();
    }


    /**
     * @brief Find valid voxel centroids within a radius of a query point using the k-d tree.
     * @details Requires `filter(true)` or `filter(output, true)` to have been called.
     * @param[in] point The query point.
     * @param[in] radius The search radius.
     * @param[out] k_leaves Vector filled with const pointers to the found valid leaves.
     * @param[out] k_sqr_distances Vector filled with squared distances to the centroids.
     * @param[in] max_nn Max number of neighbors to return (0 = unlimited).
     * @return Number of neighbors found.
     */
    int radiusSearch( const PointT& point, double radius,
                      std::vector<LeafConstPtr>& k_leaves,
                      std::vector<float>& k_sqr_distances,
                      unsigned int max_nn = 0) const
    {
        k_leaves.clear();
        k_sqr_distances.clear();

         if (!searchable_) {
            PCL_WARN("[%s::radiusSearch] Grid not searchable. Call filter(true) first.\n", getClassName().c_str());
            return 0;
        }
         if (!kdtree_.getInputCloud() || voxel_centroids_->empty()) {
             // PCL_DEBUG("[%s::radiusSearch] KdTree not initialized or no valid centroids.\n", getClassName().c_str());
             return 0; // No centroids to search
        }
        if (radius <= 0.0) return 0; // Trivial case


        std::vector<int> k_indices; // Indices relative to voxel_centroids_
        int found_k = kdtree_.radiusSearch(point, radius, k_indices, k_sqr_distances, max_nn);

        k_leaves.reserve(found_k);
        // k_sqr_distances is filled by kdtree_

        for (int centroid_idx : k_indices) {
            // Map kdtree index back to the original leaf index
             if (centroid_idx < 0 || static_cast<size_t>(centroid_idx) >= voxel_centroids_leaf_indices_.size()) {
                 PCL_ERROR("[%s::radiusSearch] Invalid index %d from KdTree.\n", getClassName().c_str(), centroid_idx);
                 continue;
             }
            size_t leaf_map_idx = voxel_centroids_leaf_indices_[centroid_idx];
            LeafConstPtr leaf_ptr = getLeaf(leaf_map_idx); // Use safe getter
            if (leaf_ptr) {
                k_leaves.push_back(leaf_ptr);
            } else {
                 // PCL_DEBUG("[%s::radiusSearch] Consistency warning: Centroid index %d maps to leaf index %zu, but getLeaf returned null.\n",
                 //           getClassName().c_str(), centroid_idx, leaf_map_idx);
            }
        }
        // Ensure distance vector matches the number of leaves actually added
        k_sqr_distances.resize(k_leaves.size());
        return k_leaves.size();
    }

    /**
     * @brief Get the voxel neighbors of a query point using DIRECT7 method (center + 6 faces).
     * @param reference_point The query point.
     * @param neighbors Output vector of const pointers to valid neighbor leaves.
     * @return Number of valid neighbors found.
     */
     int getNeighborhoodAtPoint7(const PointT& reference_point, std::vector<LeafConstPtr>& neighbors) const;

    /**
     * @brief Get the voxel neighbors of a query point using DIRECT1 method (only the center voxel).
     * @param reference_point The query point.
     * @param neighbors Output vector containing at most one const pointer to the valid leaf.
     * @return Number of valid neighbors found (0 or 1).
     */
     int getNeighborhoodAtPoint1(const PointT& reference_point, std::vector<LeafConstPtr>& neighbors) const;


protected:
    /**
     * @brief Overridden PCL VoxelGrid filter method. Computes means and covariances.
     * @details Populates the `leaves_` map with Leaf structures containing statistical
     * information for each voxel. Handles centroid calculation, covariance
     * computation, inverse covariance, eigenvalue decomposition, and
     * regularization based on `min_points_per_voxel_` and
     * `min_covar_eigvalue_mult_`.
     * @param[out] output The downsampled cloud (used internally but final centroid cloud built separately).
     */
    void applyFilter(PointCloud& output) override;

    /**
     * @brief Helper function to populate the voxel_centroids_ cloud and index map
     * based on the computed leaves_. Called internally by filter().
     */
    void buildCentroidCloudAndIndexMap();

    /**
     * @brief Helper to check if a point's coordinates are within the computed grid bounds.
     * @param p Point coordinates (uses only x, y, z).
     * @return True if the point is inside the voxel grid bounds, false otherwise.
     */
     bool isPointWithinBounds(const Eigen::Vector4f& p) const;

    // --- Member Variables ---

    bool searchable_;                    // Is k-d tree built for searching?
    int min_points_per_voxel_;         // Min points needed for a valid leaf
    double min_covar_eigvalue_mult_;   // Regularization factor for eigenvalues

    Map leaves_;                       // The core data structure: map from index to Leaf

    // Searchability support members
    PointCloudPtr voxel_centroids_;           // Cloud of valid voxel centroids
    std::vector<size_t> voxel_centroids_leaf_indices_; // Map: index in voxel_centroids_ -> index in leaves_
    pcl::KdTreeFLANN<PointT> kdtree_;         // k-d tree built on voxel_centroids_
};

} // namespace svn_ndt

// --- Implementation Inclusion ---
// The implementation needs to be included for template instantiation.
#include <voxel_grid_covariance_impl.hpp> // Include the implementation file

#endif // SVN_NDT_VOXEL_GRID_COVARIANCE_H_