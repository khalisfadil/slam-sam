/*
 * Software License Agreement (BSD License)
 * Based on PCL VoxelGridCovariance and adapted for svn_ndt.
 *
 * Point Cloud Library (PCL) - www.pointclouds.org
 * Copyright (c) 2010-2011, Willow Garage, Inc.
 * Copyright (c) 2012-, Open Perception, Inc.
 *
 * All rights reserved.
 * (License text omitted for brevity - refer to original PCL license)
 *
 * Modifications for SVN-NDT: Adapted for svn_ndt namespace, tsl::robin_map, added helpers.
 */

#ifndef SVN_NDT_VOXEL_GRID_COVARIANCE_IMPL_HPP_
#define SVN_NDT_VOXEL_GRID_COVARIANCE_IMPL_HPP_

// Include the header declaring the class
#include <voxel_grid_covariance.h> // Must include the header first

// Include necessary headers used in the implementation
#include <Eigen/Cholesky>     // For LLT decomposition if needed (not directly used here, but related)
#include <Eigen/Eigenvalues>  // For SelfAdjointEigenSolver
#include <Eigen/Dense>        // For matrix operations like inverse()

#include <pcl/common/common.h>         // For getMinMax3D
#include <pcl/common/point_tests.h>    // For pcl::isFinite
#include <pcl/filters/impl/voxel_grid.hpp> // For NdCopyPointEigenFunctor, NdCopyEigenPointFunctor
#include <pcl/io/pcd_io.h>             // For PCL_WARN_STREAM, PCL_ERROR_STREAM

#include <boost/mpl/size.hpp> // For FieldList iteration if downsample_all_data_ is true

#include <cmath>   // For std::floor, std::isfinite
#include <limits>  // For std::numeric_limits
#include <vector>
#include <map>     // Only if needed, tsl::robin_map is used internally
#include <numeric> // Potentially for std::accumulate if needed elsewhere
#include <algorithm> // Potentially for std::sort/unique if needed elsewhere

namespace svn_ndt
{

//////////////////////////////////////////////////////////////////////////////////////////
// Helper function to check if a point is within the calculated grid bounds.
template <typename PointT>
bool VoxelGridCovariance<PointT>::isPointWithinBounds(const Eigen::Vector4f& p) const
{
    // Check if the grid parameters (min_b_, max_b_, leaf_size_, inverse_leaf_size_)
    // have been computed. inverse_leaf_size_ is non-zero only after applyFilter runs successfully.
    if (this->inverse_leaf_size_[0] == 0.0f || this->inverse_leaf_size_[1] == 0.0f || this->inverse_leaf_size_[2] == 0.0f) {
        PCL_WARN("[%s::isPointWithinBounds] Grid bounds not computed yet. Call applyFilter first.\n", getClassName().c_str());
        return false; // Cannot check bounds if grid isn't set up
    }

    // Calculate the floating-point bounds derived from the integer indices min_b_ and max_b_.
    // This is generally more robust than directly comparing integer indices derived from p.
    // We add 1 to max_b_ because the index represents the *inclusive* upper bound index,
    // so the actual upper bound coordinate is (max_b_ + 1) * leaf_size_.
    float min_x_bound = static_cast<float>(this->min_b_[0]) * this->leaf_size_[0];
    float max_x_bound = static_cast<float>(this->max_b_[0] + 1) * this->leaf_size_[0];
    float min_y_bound = static_cast<float>(this->min_b_[1]) * this->leaf_size_[1];
    float max_y_bound = static_cast<float>(this->max_b_[1] + 1) * this->leaf_size_[1];
    float min_z_bound = static_cast<float>(this->min_b_[2]) * this->leaf_size_[2];
    float max_z_bound = static_cast<float>(this->max_b_[2] + 1) * this->leaf_size_[2];

    // Check if the point's coordinates fall within the [min_bound, max_bound) interval.
    // Using '<' for the upper bound is consistent with floor-based index calculation.
    return (p[0] >= min_x_bound && p[0] < max_x_bound &&
            p[1] >= min_y_bound && p[1] < max_y_bound &&
            p[2] >= min_z_bound && p[2] < max_z_bound);
}


//////////////////////////////////////////////////////////////////////////////////////////
// Main filter implementation: Calculates mean, covariance, etc. for each voxel.
template <typename PointT>
void VoxelGridCovariance<PointT>::applyFilter(PointCloud& output)
{
    // --- Standard VoxelGrid Initialization ---
    if (!this->input_) {
        PCL_WARN("[%s::applyFilter] No input dataset given!\n", getClassName().c_str());
        output.width = output.height = 0;
        output.points.clear();
        return;
    }

    // Initialize output cloud (clear points but keep header)
    output.header = this->input_->header;
    output.height = 1;
    output.is_dense = true; // Assume dense initially; might be set to false later if errors occur
    output.points.clear();

    // Determine point cloud bounds
    Eigen::Vector4f min_p, max_p;
    if (!this->filter_field_name_.empty()) {
        // Use filtered points to determine bounds
        pcl::getMinMax3D<PointT>(
            this->input_, this->filter_field_name_,
            static_cast<float>(this->filter_limit_min_), static_cast<float>(this->filter_limit_max_),
            min_p, max_p, this->filter_limit_negative_);
    } else {
        // Use all points to determine bounds
        pcl::getMinMax3D<PointT>(*this->input_, min_p, max_p);
    }

    // Check leaf size validity and prevent integer overflow
    // Calculate grid dimensions in terms of voxels
    int64_t dx = static_cast<int64_t>((max_p[0] - min_p[0]) * this->inverse_leaf_size_[0]) + 1;
    int64_t dy = static_cast<int64_t>((max_p[1] - min_p[1]) * this->inverse_leaf_size_[1]) + 1;
    int64_t dz = static_cast<int64_t>((max_p[2] - min_p[2]) * this->inverse_leaf_size_[2]) + 1;

    // Check for negative dimensions (can happen with invalid leaf size or bounds)
    if (dx < 0 || dy < 0 || dz < 0) {
        PCL_ERROR("[%s::applyFilter] Invalid leaf size or cloud bounds resulting in negative grid dimensions.\n", getClassName().c_str());
        output.width = output.height = 0; output.points.clear();
        return;
    }

    // Check for potential integer overflow when calculating 1D index
    const int64_t max_total_voxels = static_cast<int64_t>(std::numeric_limits<int32_t>::max()); // Based on PCL's internal limits
    if (dx > max_total_voxels || dy > max_total_voxels || dz > max_total_voxels || (dx*dy*dz) > max_total_voxels ) {
        PCL_ERROR("[%s::applyFilter] Leaf size is too small for the input dataset. Integer indices would overflow.\n", getClassName().c_str());
        output.width = output.height = 0; output.points.clear();
        return;
    }


    // Compute grid bounds in integer voxel indices (relative to world origin)
    this->min_b_[0] = static_cast<int>(std::floor(min_p[0] * this->inverse_leaf_size_[0]));
    this->max_b_[0] = static_cast<int>(std::floor(max_p[0] * this->inverse_leaf_size_[0]));
    this->min_b_[1] = static_cast<int>(std::floor(min_p[1] * this->inverse_leaf_size_[1]));
    this->max_b_[1] = static_cast<int>(std::floor(max_p[1] * this->inverse_leaf_size_[1]));
    this->min_b_[2] = static_cast<int>(std::floor(min_p[2] * this->inverse_leaf_size_[2]));
    this->max_b_[2] = static_cast<int>(std::floor(max_p[2] * this->inverse_leaf_size_[2]));

    // Compute divisions and multipliers for 1D index calculation
    this->div_b_ = this->max_b_ - this->min_b_ + Eigen::Vector4i::Ones();
    this->div_b_[3] = 0; // W component not used
    // divb_mul_ = [1, div_x, div_x*div_y, 0]
    this->divb_mul_ = Eigen::Vector4i(1, this->div_b_[0], this->div_b_[0] * this->div_b_[1], 0);

    // Clear the map for the new computation
    leaves_.clear();

    // --- Determine Centroid Size and Field Mappings (Only if downsample_all_data_ is true) ---
    using FieldList = typename pcl::traits::fieldList<PointT>::type;
    int centroid_size = 3; // Default to x, y, z
    std::vector<pcl::PCLPointField> fields;
    int rgba_index = -1; // Used for color packing if downsample_all_data_ is true
    if (this->downsample_all_data_) {
        centroid_size = boost::mpl::size<FieldList>::value;
        // Check for RGB/RGBA fields
        rgba_index = pcl::getFieldIndex<PointT>("rgba", fields);
        if (rgba_index == -1) {
            rgba_index = pcl::getFieldIndex<PointT>("rgb", fields);
        }
        if (rgba_index >= 0) {
            rgba_index = static_cast<int>(fields[rgba_index].offset);
            // Size adjustment handled by NdCopyPointEigenFunctor/NdCopyEigenPointFunctor
        }
    }

    // --- First Pass: Iterate through points, assign to voxels, and accumulate sums ---
    if (!this->filter_field_name_.empty()) {
        // --- Filtered Pass ---
        std::vector<pcl::PCLPointField> filter_fields;
        int filter_idx = pcl::getFieldIndex<PointT>(this->filter_field_name_, filter_fields);
        if (filter_idx == -1) {
             PCL_ERROR("[%s::applyFilter] Invalid filter field name '%s'.\n", getClassName().c_str(), this->filter_field_name_.c_str());
             output.width = output.height = 0; output.points.clear(); return;
        }
        const auto& filter_field = filter_fields[filter_idx];
        const float filter_min = static_cast<float>(this->filter_limit_min_);
        const float filter_max = static_cast<float>(this->filter_limit_max_);

        for (const auto& point : this->input_->points) {
            if (!pcl::isFinite(point)) continue; // Skip invalid points (NaN, Inf)

            // Check filter field value
            const auto* pt_data = reinterpret_cast<const std::uint8_t*>(&point);
            float field_value = 0;
            memcpy(&field_value, pt_data + filter_field.offset, sizeof(float));
            bool point_passes_filter = this->filter_limit_negative_ ?
                                       (field_value < filter_max && field_value > filter_min) : // Keep points inside the interval
                                       (field_value > filter_max || field_value < filter_min);  // Keep points outside the interval
            if (!point_passes_filter) continue;

            // Calculate voxel index
            int ijk0 = static_cast<int>(std::floor(point.x * this->inverse_leaf_size_[0]) - this->min_b_[0]);
            int ijk1 = static_cast<int>(std::floor(point.y * this->inverse_leaf_size_[1]) - this->min_b_[1]);
            int ijk2 = static_cast<int>(std::floor(point.z * this->inverse_leaf_size_[2]) - this->min_b_[2]);
            size_t idx = static_cast<size_t>(ijk0 * this->divb_mul_[0] + ijk1 * this->divb_mul_[1] + ijk2 * this->divb_mul_[2]);

            // Get or create the leaf (operator[] creates if not present)
            Leaf& leaf = leaves_[idx];

            // Initialize Nd centroid if needed (first point in this leaf)
            if (leaf.nr_points_ == 0 && this->downsample_all_data_) {
                leaf.centroid_.resize(centroid_size);
                leaf.centroid_.setZero();
            }

            // Accumulate sums for mean and covariance (single-pass method)
            Eigen::Vector3d pt3d(point.x, point.y, point.z);
            leaf.mean_ += pt3d; // Sum(xi)
            leaf.cov_ += pt3d * pt3d.transpose(); // Sum(xi * xi^T)

            // Accumulate Nd centroid if needed
            if (this->downsample_all_data_) {
                 Eigen::VectorXf pt_centroid = Eigen::VectorXf::Zero(centroid_size);
                 pcl::for_each_type<FieldList>(pcl::NdCopyPointEigenFunctor<PointT>(point, pt_centroid));
                 leaf.centroid_ += pt_centroid;
            }
            leaf.nr_points_++; // Increment point count for this leaf
        }
    } else {
        // --- Unfiltered Pass (Process all points) ---
        for (const auto& point : this->input_->points) {
            if (!pcl::isFinite(point)) continue;

            // Calculate voxel index
            int ijk0 = static_cast<int>(std::floor(point.x * this->inverse_leaf_size_[0]) - this->min_b_[0]);
            int ijk1 = static_cast<int>(std::floor(point.y * this->inverse_leaf_size_[1]) - this->min_b_[1]);
            int ijk2 = static_cast<int>(std::floor(point.z * this->inverse_leaf_size_[2]) - this->min_b_[2]);
            size_t idx = static_cast<size_t>(ijk0 * this->divb_mul_[0] + ijk1 * this->divb_mul_[1] + ijk2 * this->divb_mul_[2]);

            // Get or create the leaf
            Leaf& leaf = leaves_[idx];

            // Initialize Nd centroid if needed
            if (leaf.nr_points_ == 0 && this->downsample_all_data_) {
                leaf.centroid_.resize(centroid_size);
                leaf.centroid_.setZero();
            }

            // Accumulate sums
            Eigen::Vector3d pt3d(point.x, point.y, point.z);
            leaf.mean_ += pt3d;
            leaf.cov_ += pt3d * pt3d.transpose();

            // Accumulate Nd centroid if needed
            if (this->downsample_all_data_) {
                 Eigen::VectorXf pt_centroid = Eigen::VectorXf::Zero(centroid_size);
                 pcl::for_each_type<FieldList>(pcl::NdCopyPointEigenFunctor<PointT>(point, pt_centroid));
                 leaf.centroid_ += pt_centroid;
            }
            leaf.nr_points_++;
        }
    }

    // --- Second Pass: Finalize calculations (mean, covariance, inverse, eigen decomposition) ---
    output.points.reserve(leaves_.size()); // Reserve space roughly proportional to the number of populated leaves

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver; // Reusable solver
    bool numerical_error_detected = false; // Track if any leaves are discarded due to errors

    // Create a list of keys to iterate over, allowing safe deletion from the map during iteration
    std::vector<size_t> keys_to_process;
    keys_to_process.reserve(leaves_.size());
    for(const auto& pair : leaves_) {
        keys_to_process.push_back(pair.first);
    }

    // Process each potentially valid leaf
    for (const size_t index : keys_to_process) {
        // Access the leaf directly using operator[]. Safe because we iterate existing keys.
        Leaf& leaf = leaves_[index];

        // --- Voxel Validity Check ---
        if (leaf.nr_points_ < min_points_per_voxel_) {
            leaves_.erase(index); // Erase leaf if it doesn't have enough points
            continue; // Skip further processing for this leaf
        }

        // --- Calculate Mean ---
        const double point_count = static_cast<double>(leaf.nr_points_);
        const Eigen::Vector3d pt_sum = leaf.mean_; // Store sum before dividing
        leaf.mean_ /= point_count; // E[x] = Sum(xi) / N

        // --- Calculate Nd Centroid (if needed) ---
        if (this->downsample_all_data_) {
            leaf.centroid_ /= static_cast<float>(point_count);
        }

        // --- Calculate Covariance (using single-pass results) ---
        // Cov = E[xx^T] - E[x]E[x]^T = (Sum(xi*xi^T) / N) - mean * mean^T
        leaf.cov_ = (leaf.cov_ / point_count) - (leaf.mean_ * leaf.mean_.transpose());
        // Apply Bessel's correction for sample covariance: multiply by N / (N-1)
        if (point_count > 1) {
             leaf.cov_ *= (point_count / (point_count - 1.0));
        } else {
             // Handle N=1 case: Covariance is undefined, set to small identity for numerical stability if needed
             leaf.cov_.setIdentity();
             leaf.cov_ *= 1e-9; // Or some other small value
        }

        // --- Eigen Decomposition and Regularization ---
        eigensolver.compute(leaf.cov_);
        Eigen::Vector3d eigen_values = eigensolver.eigenvalues(); // Eigenvalues are sorted ascending
        leaf.evecs_ = eigensolver.eigenvectors(); // Eigenvectors form columns

        // Check for numerical issues (non-positive semi-definite matrix)
        const double min_eigenvalue_threshold = 1e-12; // Absolute minimum threshold
        if (eigen_values(0) < 0 || eigen_values(1) < 0 || eigen_values(2) < min_eigenvalue_threshold) {
            // PCL_DEBUG("[%s::applyFilter] Voxel %zu has non-positive definite covariance. Discarding.\n", getClassName().c_str(), index);
            numerical_error_detected = true;
            leaves_.erase(index);
            continue;
        }

        // Apply regularization: Inflate small eigenvalues relative to the largest one
        double max_eigenvalue = eigen_values(2); // Largest eigenvalue
        double min_acceptable_eigenvalue = std::max(min_eigenvalue_threshold, max_eigenvalue * min_covar_eigvalue_mult_);

        bool needs_recomposition = false;
        if (eigen_values(0) < min_acceptable_eigenvalue) {
            eigen_values(0) = min_acceptable_eigenvalue;
            needs_recomposition = true;
        }
        if (eigen_values(1) < min_acceptable_eigenvalue) {
            eigen_values(1) = min_acceptable_eigenvalue;
            needs_recomposition = true;
        }
        // eigen_values(2) is implicitly handled if it was below min_eigenvalue_threshold earlier

        leaf.evals_ = eigen_values; // Store the (potentially inflated) eigenvalues

        // Reconstruct covariance matrix if eigenvalues were inflated
        if (needs_recomposition) {
            leaf.cov_ = leaf.evecs_ * leaf.evals_.asDiagonal() * leaf.evecs_.transpose();
        }

        // --- Calculate Inverse Covariance ---
        leaf.icov_ = leaf.cov_.inverse();

        // Check inverse stability
        const double max_inverse_coeff = 1e12; // Heuristic threshold for instability
        if (!leaf.icov_.allFinite() || leaf.icov_.cwiseAbs().maxCoeff() > max_inverse_coeff) {
           // PCL_DEBUG("[%s::applyFilter] Voxel %zu inverse covariance is unstable. Discarding.\n", getClassName().c_str(), index);
            numerical_error_detected = true;
            leaves_.erase(index);
            continue;
        }

        // --- Add Centroid to Output Cloud (Only if all checks passed) ---
        PointT downsampled_pt;
        if (this->downsample_all_data_) {
            // Copy averaged Nd centroid vector into the point struct fields using PCL helper
            pcl::for_each_type<FieldList>(pcl::NdCopyEigenPointFunctor<PointT>(leaf.centroid_, downsampled_pt));

            // Manually handle RGB packing if necessary (based on PCL's VoxelGrid implementation)
            if (rgba_index >= 0 && centroid_size >= 3) {
                 float r = leaf.centroid_[centroid_size - 3];
                 float g = leaf.centroid_[centroid_size - 2];
                 float b = leaf.centroid_[centroid_size - 1];
                 std::uint32_t rgb = (static_cast<std::uint32_t>(r) << 16 |
                                      static_cast<std::uint32_t>(g) << 8  |
                                      static_cast<std::uint32_t>(b));
                 memcpy(reinterpret_cast<char*>(&downsampled_pt) + rgba_index, &rgb, sizeof(std::uint32_t));
            }
        } else {
            // Standard NDT case: Output point is the 3D mean
            downsampled_pt.x = static_cast<float>(leaf.mean_.x());
            downsampled_pt.y = static_cast<float>(leaf.mean_.y());
            downsampled_pt.z = static_cast<float>(leaf.mean_.z());

            // --- OPTIONAL: Handle other fields like intensity ---
            // --- END OPTIONAL ---
        }

        output.points.push_back(downsampled_pt);

    } // End loop through leaf keys

    // Finalize output cloud properties
    output.width = static_cast<std::uint32_t>(output.points.size());
    output.is_dense = !numerical_error_detected; // Cloud is not dense if any voxels were discarded due to errors

} // End applyFilter


//////////////////////////////////////////////////////////////////////////////////////////
// Helper function to build the cloud of valid centroids and the index map
// used for KdTree searching.
template <typename PointT>
void VoxelGridCovariance<PointT>::buildCentroidCloudAndIndexMap()
{
    // Ensure the internal centroid cloud pointer is valid
    if (!voxel_centroids_) {
        voxel_centroids_.reset(new PointCloud());
    }

    // Clear previous data
    voxel_centroids_->clear();
    voxel_centroids_leaf_indices_.clear();

    // If no valid leaves were found after filtering, initialize empty cloud and return
    if (leaves_.empty()) {
        voxel_centroids_->width = 0;
        voxel_centroids_->height = 1;
        voxel_centroids_->is_dense = true;
        return;
    }

    // Reserve space for efficiency
    voxel_centroids_->reserve(leaves_.size());
    voxel_centroids_leaf_indices_.reserve(leaves_.size());

    PointT temp_pt; // Reusable point for efficiency

    // Iterate through the *filtered* leaves_ map (contains only valid voxels)
    for (const auto& pair : leaves_) {
        const size_t original_index = pair.first; // The key in the leaves_ map
        const Leaf& leaf = pair.second;

        // Note: The check leaf.getPointCount() >= min_points_per_voxel_ is implicitly
        // satisfied here because applyFilter already removed invalid leaves from the map.

        // Create the centroid point
        temp_pt.x = static_cast<float>(leaf.getMean().x());
        temp_pt.y = static_cast<float>(leaf.getMean().y());
        temp_pt.z = static_cast<float>(leaf.getMean().z());
        // --- OPTIONAL: Copy other fields (e.g., average intensity) ---
        // --- END OPTIONAL ---

        // Add point to the cloud and store the original leaf index
        voxel_centroids_->push_back(temp_pt);
        voxel_centroids_leaf_indices_.push_back(original_index);
    }

    // Finalize centroid cloud properties
    voxel_centroids_->width = static_cast<std::uint32_t>(voxel_centroids_->size());
    voxel_centroids_->height = 1;
    voxel_centroids_->is_dense = true; // Centroid cloud itself should always be dense
}


//////////////////////////////////////////////////////////////////////////////////////////
// Neighbor search implementations

// --- KdTree-based searches (require searchable_ == true) ---

template <typename PointT>
int VoxelGridCovariance<PointT>::nearestKSearch(
    const PointT& point, int k,
    std::vector<LeafConstPtr>& k_leaves,
    std::vector<float>& k_sqr_distances) const
{
    k_leaves.clear();
    k_sqr_distances.clear();

    if (!searchable_) {
        PCL_WARN("[%s::nearestKSearch] Grid not searchable. Call filter(true) or filter(output, true) first.\n", getClassName().c_str());
        return 0;
    }
    if (!kdtree_.getInputCloud() || !voxel_centroids_ || voxel_centroids_->empty()) {
         // PCL_DEBUG("[%s::nearestKSearch] KdTree not initialized or no valid centroids to search.\n", getClassName().c_str());
         return 0; // No centroids to search
    }
    if (k <= 0) {
        return 0; // Trivial case
    }


    std::vector<int> k_indices; // Indices relative to voxel_centroids_ cloud
    k_indices.reserve(k);       // Reserve space
    k_sqr_distances.reserve(k);

    // Perform the KdTree search on the centroid cloud
    int found_k = kdtree_.nearestKSearch(point, k, k_indices, k_sqr_distances);

    k_leaves.reserve(found_k); // Reserve final size

    // Map KdTree indices back to original Leaf indices and get Leaf pointers
    for (int centroid_idx : k_indices) {
        // Basic validity check on the index returned by KdTree
        if (centroid_idx < 0 || static_cast<size_t>(centroid_idx) >= voxel_centroids_leaf_indices_.size()) {
             PCL_ERROR("[%s::nearestKSearch] Invalid index %d received from KdTree search.\n", getClassName().c_str(), centroid_idx);
             continue; // Skip this invalid index
        }

        // Look up the original leaf index using the mapping vector
        size_t leaf_map_idx = voxel_centroids_leaf_indices_[centroid_idx];

        // Retrieve the leaf using the safe getter (includes point count check)
        LeafConstPtr leaf_ptr = getLeaf(leaf_map_idx);
        if (leaf_ptr) {
            k_leaves.push_back(leaf_ptr);
        } else {
             // This case should ideally not happen if buildCentroidCloudAndIndexMap is correct,
             // but could occur if leaves_ was modified after building the map/kdtree.
             // PCL_DEBUG("[%s::nearestKSearch] Consistency warning: Centroid index %d maps to leaf index %zu, but getLeaf returned null.\n",
             //           getClassName().c_str(), centroid_idx, leaf_map_idx);
        }
    }

    // Ensure the distances vector matches the number of leaves actually added
    // (in case some indices were invalid or getLeaf failed)
    k_sqr_distances.resize(k_leaves.size());
    return static_cast<int>(k_leaves.size());
}


template <typename PointT>
int VoxelGridCovariance<PointT>::radiusSearch(
    const PointT& point, double radius,
    std::vector<LeafConstPtr>& k_leaves,
    std::vector<float>& k_sqr_distances,
    unsigned int max_nn) const
{
    k_leaves.clear();
    k_sqr_distances.clear();

    if (!searchable_) {
        PCL_WARN("[%s::radiusSearch] Grid not searchable. Call filter(true) or filter(output, true) first.\n", getClassName().c_str());
        return 0;
    }
     if (!kdtree_.getInputCloud() || !voxel_centroids_ || voxel_centroids_->empty()) {
         // PCL_DEBUG("[%s::radiusSearch] KdTree not initialized or no valid centroids to search.\n", getClassName().c_str());
         return 0; // No centroids to search
    }
    if (radius <= 0.0) {
        return 0; // Trivial case
    }

    std::vector<int> k_indices; // Indices relative to voxel_centroids_ cloud
    // Reserve estimated space? Difficult for radius search.

    // Perform the KdTree search on the centroid cloud
    int found_k = kdtree_.radiusSearch(point, radius, k_indices, k_sqr_distances, max_nn);

    k_leaves.reserve(found_k); // Reserve final size

    // Map KdTree indices back to original Leaf indices and get Leaf pointers
    for (int centroid_idx : k_indices) {
        if (centroid_idx < 0 || static_cast<size_t>(centroid_idx) >= voxel_centroids_leaf_indices_.size()) {
             PCL_ERROR("[%s::radiusSearch] Invalid index %d received from KdTree search.\n", getClassName().c_str(), centroid_idx);
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

    // Ensure the distances vector matches the number of leaves actually added
    k_sqr_distances.resize(k_leaves.size());
    return static_cast<int>(k_leaves.size());
}


// --- Direct Neighbor Searches ---

template <typename PointT>
int VoxelGridCovariance<PointT>::getNeighborhoodAtPoint7(
    const PointT& reference_point, std::vector<LeafConstPtr>& neighbors) const
{
    neighbors.clear();
    neighbors.reserve(7); // Reserve space for center + 6 potential face neighbors

    // Get the leaf containing the reference point first
    LeafConstPtr center_leaf = getLeaf(reference_point);
    if (center_leaf) {
        neighbors.push_back(center_leaf);
    }

    // Check the 6 face-adjacent neighbors
    Eigen::Vector3f p = reference_point.getVector3fMap();
    float wx = this->leaf_size_[0];
    float wy = this->leaf_size_[1];
    float wz = this->leaf_size_[2];

    // Proceed only if leaf size is valid (non-zero)
    if (wx <= 0 || wy <= 0 || wz <= 0) {
        if (!center_leaf) { // Only warn if center wasn't found either AND leaf size is bad
            PCL_WARN("[%s::getNeighborhoodAtPoint7] Leaf size is non-positive, cannot check neighbors.\n", getClassName().c_str());
        }
        return static_cast<int>(neighbors.size());
    }

    // Check neighbors using getLeaf(Vector3f), which handles bounds checking and validity
    LeafConstPtr neighbor = nullptr;
    neighbor = getLeaf(Eigen::Vector3f(p.x() + wx, p.y(),     p.z()));     if (neighbor) neighbors.push_back(neighbor);
    neighbor = getLeaf(Eigen::Vector3f(p.x() - wx, p.y(),     p.z()));     if (neighbor) neighbors.push_back(neighbor);
    neighbor = getLeaf(Eigen::Vector3f(p.x(),      p.y() + wy, p.z()));     if (neighbor) neighbors.push_back(neighbor);
    neighbor = getLeaf(Eigen::Vector3f(p.x(),      p.y() - wy, p.z()));     if (neighbor) neighbors.push_back(neighbor);
    neighbor = getLeaf(Eigen::Vector3f(p.x(),      p.y(),     p.z() + wz)); if (neighbor) neighbors.push_back(neighbor);
    neighbor = getLeaf(Eigen::Vector3f(p.x(),      p.y(),     p.z() - wz)); if (neighbor) neighbors.push_back(neighbor);

    // Optional: Remove duplicates if a point lies exactly on a boundary
    // std::sort(neighbors.begin(), neighbors.end());
    // neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

    return static_cast<int>(neighbors.size());
}


template <typename PointT>
int VoxelGridCovariance<PointT>::getNeighborhoodAtPoint1(
    const PointT& reference_point, std::vector<LeafConstPtr>& neighbors) const
{
    neighbors.clear();
    // Get only the single leaf containing the reference point
    LeafConstPtr leaf = getLeaf(reference_point);
    if (leaf) {
        neighbors.push_back(leaf);
        return 1;
    }
    return 0; // No valid leaf found containing the point
}


} // namespace svn_ndt

#endif // SVN_NDT_VOXEL_GRID_COVARIANCE_IMPL_HPP_