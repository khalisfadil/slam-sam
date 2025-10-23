/*
 * Software License Agreement (BSD License)
 *
 * Point Cloud Library (PCL) - www.pointclouds.org
 * Copyright (c) 2010-2011, Willow Garage, Inc.
 * Copyright (c) 2012-, Open Perception, Inc.
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
 * Modifications for SVN-NDT: Adapted for svn_ndt namespace, tsl::robin_map, added helpers.
 */

#ifndef SVN_NDT_VOXEL_GRID_COVARIANCE_IMPL_HPP_
#define SVN_NDT_VOXEL_GRID_COVARIANCE_IMPL_HPP_

// Include the header declaring the class
#include <voxel_grid_covariance.h>

// Include necessary headers used in the implementation
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>

#include <pcl/common/common.h>
#include <pcl/filters/boost.h>
#include <pcl/common/point_tests.h> // For isFinite

#include <cmath>
#include <limits>
#include <vector>
#include <map>

// Include NdCopyPointEigenFunctor definition if needed
// Usually in <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/filters/impl/voxel_grid.hpp>


namespace svn_ndt
{

//////////////////////////////////////////////////////////////////////////////////////////
// Helper function to check bounds
template <typename PointT>
bool VoxelGridCovariance<PointT>::isPointWithinBounds(const Eigen::Vector4f& p) const
{
    // Check if the grid bounds (min_b_, max_b_) have been computed
    // inverse_leaf_size_ is non-zero only after applyFilter computes bounds
     if (this->inverse_leaf_size_[0] == 0.0f || this->inverse_leaf_size_[1] == 0.0f || this->inverse_leaf_size_[2] == 0.0f) {
         PCL_WARN("[%s::isPointWithinBounds] Grid bounds not computed yet. Call applyFilter first.\n", getClassName().c_str());
         return false; // Cannot check bounds if grid isn't set up
     }
     
     // Note: min_b_ and max_b_ are integer indices, but they are derived from
     // floating point min_p/max_p. A point p should satisfy:
     // floor(p[i] * inv_leaf[i]) >= min_b_[i] AND floor(p[i] * inv_leaf[i]) <= max_b_[i]
     // This is equivalent to checking against the original float bounds derived from min_p/max_p
     // which VoxelGrid base class stores implicitly (or recalculates in getMinMax).
     // Let's use the float bounds derived from min_b_/max_b_ for robustness.

     // Calculate float bounds corresponding to the integer indices min_b_, max_b_
     float min_x_bound = static_cast<float>(this->min_b_[0]) * this->leaf_size_[0];
     float max_x_bound = static_cast<float>(this->max_b_[0] + 1) * this->leaf_size_[0]; // +1 because max_b_ is inclusive index
     float min_y_bound = static_cast<float>(this->min_b_[1]) * this->leaf_size_[1];
     float max_y_bound = static_cast<float>(this->max_b_[1] + 1) * this->leaf_size_[1];
     float min_z_bound = static_cast<float>(this->min_b_[2]) * this->leaf_size_[2];
     float max_z_bound = static_cast<float>(this->max_b_[2] + 1) * this->leaf_size_[2];

     return (p[0] >= min_x_bound && p[0] < max_x_bound && // Use < for upper bound consistency
             p[1] >= min_y_bound && p[1] < max_y_bound &&
             p[2] >= min_z_bound && p[2] < max_z_bound);
}


//////////////////////////////////////////////////////////////////////////////////////////
// Main filter implementation: Calculates mean, covariance, etc.
template <typename PointT>
void VoxelGridCovariance<PointT>::applyFilter(PointCloud& output)
{
    // --- Standard VoxelGrid Initialization ---
    if (!input_)
    {
        PCL_WARN("[%s::applyFilter] No input dataset given!\n", getClassName().c_str());
        output.width = output.height = 0;
        output.points.clear();
        return;
    }

    output.header = input_->header;
    output.height = 1;
    output.is_dense = true; // Assume dense initially
    output.points.clear(); // Clear output points, retain header info

    Eigen::Vector4f min_p, max_p;
    if (!filter_field_name_.empty())
    {
        pcl::getMinMax3D<PointT>(
            input_, filter_field_name_,
            static_cast<float>(filter_limit_min_), static_cast<float>(filter_limit_max_),
            min_p, max_p, filter_limit_negative_);
    }
    else
    {
        pcl::getMinMax3D<PointT>(*input_, min_p, max_p);
    }

    // Check leaf size validity
    int64_t dx = static_cast<int64_t>((max_p[0] - min_p[0]) * inverse_leaf_size_[0]) + 1;
    int64_t dy = static_cast<int64_t>((max_p[1] - min_p[1]) * inverse_leaf_size_[1]) + 1;
    int64_t dz = static_cast<int64_t>((max_p[2] - min_p[2]) * inverse_leaf_size_[2]) + 1;

    if (dx < 0 || dy < 0 || dz < 0) {
        PCL_ERROR("[%s::applyFilter] Invalid leaf size or cloud bounds resulting in negative grid dimensions.\n", getClassName().c_str());
        output.clear();
        return;
    }

    if ((dx * dy * dz) > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
    {
        PCL_ERROR("[%s::applyFilter] Leaf size is too small for the input dataset. Integer indices would overflow.\n", getClassName().c_str());
        output.clear();
        return;
    }

    // Compute grid bounds in voxel indices (relative to world origin)
    min_b_[0] = static_cast<int>(std::floor(min_p[0] * inverse_leaf_size_[0]));
    max_b_[0] = static_cast<int>(std::floor(max_p[0] * inverse_leaf_size_[0]));
    min_b_[1] = static_cast<int>(std::floor(min_p[1] * inverse_leaf_size_[1]));
    max_b_[1] = static_cast<int>(std::floor(max_p[1] * inverse_leaf_size_[1]));
    min_b_[2] = static_cast<int>(std::floor(min_p[2] * inverse_leaf_size_[2]));
    max_b_[2] = static_cast<int>(std::floor(max_p[2] * inverse_leaf_size_[2]));

    // Compute divisions and multipliers
    div_b_ = max_b_ - min_b_ + Eigen::Vector4i::Ones();
    div_b_[3] = 0;
    divb_mul_ = Eigen::Vector4i(1, div_b_[0], div_b_[0] * div_b_[1], 0);

    leaves_.clear(); // Clear map for new computation

    // --- Centroid Size and Field Mappings (As per original impl) ---
    using FieldList = typename pcl::traits::fieldList<PointT>::type;
    int centroid_size = 3; // Start with x, y, z
    if (downsample_all_data_) {
        centroid_size = pcl::getFieldList<PointT>().size();
    }
    std::vector<pcl::PCLPointField> fields;
    int rgba_index = -1;
    if (downsample_all_data_) {
        rgba_index = pcl::getFieldIndex<PointT>("rgba", fields); // Prefer rgba
        if (rgba_index == -1) {
            rgba_index = pcl::getFieldIndex<PointT>("rgb", fields);
        }
        if (rgba_index >= 0) {
            rgba_index = static_cast<int>(fields[rgba_index].offset);
            // Size adjustment handled by NdCopyPointEigenFunctor usually
        }
    }

    // --- First Pass: Populate Leaves ---
    // (Identical logic to the original implementation you provided)
    if (!filter_field_name_.empty())
    {
        std::vector<pcl::PCLPointField> filter_fields;
        int filter_idx = pcl::getFieldIndex<PointT>(filter_field_name_, filter_fields);
        if (filter_idx == -1) {
             PCL_ERROR("[%s::applyFilter] Invalid filter field name '%s'.\n", getClassName().c_str(), filter_field_name_.c_str());
             output.clear(); return;
        }
        const auto& filter_field = filter_fields[filter_idx];

        for (const auto& point : input_->points)
        {
            if (!pcl::isFinite(point)) continue;
            const auto* pt_data = reinterpret_cast<const std::uint8_t*>(&point);
            float field_value = 0;
            memcpy(&field_value, pt_data + filter_field.offset, sizeof(float));
            if (filter_limit_negative_ ? (field_value < filter_limit_max_ && field_value > filter_limit_min_) : (field_value > filter_limit_max_ || field_value < filter_limit_min_)) continue;

            int ijk0 = static_cast<int>(std::floor(point.x * inverse_leaf_size_[0]) - min_b_[0]);
            int ijk1 = static_cast<int>(std::floor(point.y * inverse_leaf_size_[1]) - min_b_[1]);
            int ijk2 = static_cast<int>(std::floor(point.z * inverse_leaf_size_[2]) - min_b_[2]);
            size_t idx = static_cast<size_t>(ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2]);
            Leaf& leaf = leaves_[idx]; // Creates leaf if it doesn't exist

            if (leaf.nr_points_ == 0 && downsample_all_data_) {
                leaf.centroid_.resize(centroid_size); leaf.centroid_.setZero();
            }
            Eigen::Vector3d pt3d(point.x, point.y, point.z);
            leaf.mean_ += pt3d;
            leaf.cov_ += pt3d * pt3d.transpose();
            if (downsample_all_data_) {
                 Eigen::VectorXf pt_centroid = Eigen::VectorXf::Zero(centroid_size);
                 pcl::for_each_type<FieldList>(pcl::NdCopyPointEigenFunctor<PointT>(point, pt_centroid));
                 leaf.centroid_ += pt_centroid;
            }
            leaf.nr_points_++;
        }
    }
    else // Process all points
    {
        for (const auto& point : input_->points)
        {
            if (!pcl::isFinite(point)) continue;
            int ijk0 = static_cast<int>(std::floor(point.x * inverse_leaf_size_[0]) - min_b_[0]);
            int ijk1 = static_cast<int>(std::floor(point.y * inverse_leaf_size_[1]) - min_b_[1]);
            int ijk2 = static_cast<int>(std::floor(point.z * inverse_leaf_size_[2]) - min_b_[2]);
            size_t idx = static_cast<size_t>(ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2]);
            Leaf& leaf = leaves_[idx];

            if (leaf.nr_points_ == 0 && downsample_all_data_) {
                leaf.centroid_.resize(centroid_size); leaf.centroid_.setZero();
            }
            Eigen::Vector3d pt3d(point.x, point.y, point.z);
            leaf.mean_ += pt3d;
            leaf.cov_ += pt3d * pt3d.transpose();
            if (downsample_all_data_) {
                 Eigen::VectorXf pt_centroid = Eigen::VectorXf::Zero(centroid_size);
                 pcl::for_each_type<FieldList>(pcl::NdCopyPointEigenFunctor<PointT>(point, pt_centroid));
                 leaf.centroid_ += pt_centroid;
            }
            leaf.nr_points_++;
        }
    }

    // --- Second Pass: Finalize Calculations ---
    output.points.reserve(leaves_.size()); // Reserve estimated size

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver;
    bool numerical_error = false;

    for (auto it = leaves_.begin(); it != leaves_.end(); /* no increment here */ ) // Use iterator loop to allow erasing invalid leaves
    {
        Leaf& leaf = it->second;
        const size_t index = it->first; // Store index before potential erase

        if (leaf.nr_points_ < min_points_per_voxel_) {
            it = leaves_.erase(it); // Erase invalid leaves from the map
            continue; // Skip rest of processing for this leaf
        }

        const double point_count = static_cast<double>(leaf.nr_points_);
        Eigen::Vector3d pt_sum = leaf.mean_;
        leaf.mean_ /= point_count;

        if (downsample_all_data_) {
            leaf.centroid_ /= static_cast<float>(point_count);
        }

        leaf.cov_ = (leaf.cov_ / point_count) - (leaf.mean_ * leaf.mean_.transpose());
        if (point_count > 1) {
             leaf.cov_ *= (point_count / (point_count - 1.0)); // Sample covariance
        } else {
             leaf.cov_.setIdentity(); leaf.cov_ *= 1e-9; // Handle N=1 case
        }

        eigensolver.compute(leaf.cov_);
        Eigen::Vector3d eigen_values = eigensolver.eigenvalues();
        leaf.evecs_ = eigensolver.eigenvectors();

        if (eigen_values(0) < 0 || eigen_values(1) < 0 || eigen_values(2) <= 1e-12) // Use tolerance
        {
            // PCL_DEBUG("[%s::applyFilter] Voxel %zu has non-positive definite cov.\n", getClassName().c_str(), index);
            numerical_error = true;
            it = leaves_.erase(it); // Erase invalid leaf
            continue;
        }

        double max_eigenvalue = eigen_values(2); // Sorted ascending
        double min_acceptable_eigenvalue = max_eigenvalue * min_covar_eigvalue_mult_;
        if (min_acceptable_eigenvalue < 1e-12) min_acceptable_eigenvalue = 1e-12; // Absolute minimum

        bool needs_recomposition = false;
        if (eigen_values(0) < min_acceptable_eigenvalue) {
            eigen_values(0) = min_acceptable_eigenvalue; needs_recomposition = true;
        }
        if (eigen_values(1) < min_acceptable_eigenvalue) {
            eigen_values(1) = min_acceptable_eigenvalue; needs_recomposition = true;
        }
        // Eigenvalue 2 is implicitly handled if it was <= 1e-12 earlier

        leaf.evals_ = eigen_values;

        if (needs_recomposition) {
            leaf.cov_ = leaf.evecs_ * leaf.evals_.asDiagonal() * leaf.evecs_.transpose();
             // Recompute eigensolver if needed for consistent evecs_? Usually not necessary.
             // eigensolver.compute(leaf.cov_);
             // leaf.evecs_ = eigensolver.eigenvectors(); // Update evecs_ based on regularized cov_
        }

        leaf.icov_ = leaf.cov_.inverse();

        // Check inverse stability again
        if (!leaf.icov_.allFinite() || leaf.icov_.cwiseAbs().maxCoeff() > 1e12)
        {
           // PCL_DEBUG("[%s::applyFilter] Voxel %zu inverse cov unstable.\n", getClassName().c_str(), index);
            numerical_error = true;
            it = leaves_.erase(it); // Erase invalid leaf
            continue;
        }

        // --- Add centroid to output cloud (only for valid leaves) ---
        PointT downsampled_pt;
        if (downsample_all_data_) {
            pcl::for_each_type<FieldList>(pcl::NdCopyEigenPointFunctor<PointT>(leaf.centroid_, downsampled_pt));
            if (rgba_index >= 0) {
                 // Packing logic (ensure NdCopyEigenPointFunctor doesn't already do this)
                 // This assumes centroid stores r,g,b as last 3 floats
                 // Verify this assumption based on PCL source or testing
                 if (centroid_size >= 3) {
                     float r = leaf.centroid_[centroid_size - 3];
                     float g = leaf.centroid_[centroid_size - 2];
                     float b = leaf.centroid_[centroid_size - 1];
                     std::uint32_t rgb = (static_cast<std::uint32_t>(r) << 16 |
                                          static_cast<std::uint32_t>(g) << 8  |
                                          static_cast<std::uint32_t>(b));
                     memcpy(reinterpret_cast<char*>(&downsampled_pt) + rgba_index, &rgb, sizeof(std::uint32_t));
                 }
            }
        } else {
            downsampled_pt.x = static_cast<float>(leaf.mean_.x());
            downsampled_pt.y = static_cast<float>(leaf.mean_.y());
            downsampled_pt.z = static_cast<float>(leaf.mean_.z());
            // Copy other fields if needed and available (e.g., intensity average)
        }
        output.points.push_back(downsampled_pt);

        ++it; // Move to the next element only if current one wasn't erased

    } // End loop through leaves

    output.width = static_cast<std::uint32_t>(output.points.size());
    output.is_dense = !numerical_error; // Cloud is not dense if errors occurred

} // End applyFilter


//////////////////////////////////////////////////////////////////////////////////////////
// Helper to build centroid cloud and index map (used by filter overloads)
template <typename PointT>
void VoxelGridCovariance<PointT>::buildCentroidCloudAndIndexMap()
{
    // Ensure internal cloud is allocated
    if (!voxel_centroids_) voxel_centroids_.reset(new PointCloud());

    voxel_centroids_->clear();
    voxel_centroids_leaf_indices_.clear();

    if (leaves_.empty()) {
        voxel_centroids_->width = 0;
        voxel_centroids_->height = 1;
        voxel_centroids_->is_dense = true;
        return; // Nothing to do if leaves_ is empty
    }

    voxel_centroids_->reserve(leaves_.size()); // Reserve estimated size
    voxel_centroids_leaf_indices_.reserve(leaves_.size());

    PointT temp_pt; // Reusable point

    // This loop iterates over the *potentially filtered* leaves map from applyFilter
    for (const auto& pair : leaves_) {
        const size_t index = pair.first;
        const Leaf& leaf = pair.second;

        // Only add centroids for leaves that were deemed valid in applyFilter
        if (leaf.getPointCount() >= min_points_per_voxel_) {
             temp_pt.x = static_cast<float>(leaf.getMean().x());
             temp_pt.y = static_cast<float>(leaf.getMean().y());
             temp_pt.z = static_cast<float>(leaf.getMean().z());
             // Copy other fields if needed (intensity etc.)
             // Needs logic similar to applyFilter's output population

             voxel_centroids_->push_back(temp_pt);
             voxel_centroids_leaf_indices_.push_back(index); // Store the original leaf map index
        }
    }
    voxel_centroids_->width = voxel_centroids_->size();
    voxel_centroids_->height = 1;
    voxel_centroids_->is_dense = true; // Centroid cloud itself should be dense
}


//////////////////////////////////////////////////////////////////////////////////////////
// Neighbor search implementations
template <typename PointT>
int VoxelGridCovariance<PointT>::getNeighborhoodAtPoint7(
    const PointT& reference_point, std::vector<LeafConstPtr>& neighbors) const
{
    neighbors.clear();
    neighbors.reserve(7); // Reserve space for center + 6 faces

    // Get center leaf first
    LeafConstPtr center_leaf = getLeaf(reference_point);
    if (center_leaf) {
        neighbors.push_back(center_leaf);
    }

    // Check 6 face neighbors
    Eigen::Vector3f p = reference_point.getVector3fMap();
    float wx = leaf_size_[0];
    float wy = leaf_size_[1];
    float wz = leaf_size_[2];

    // Check neighbors only if leaf size is valid
    if (wx <= 0 || wy <= 0 || wz <= 0) return neighbors.size();

    LeafConstPtr neighbor;
    neighbor = getLeaf(Eigen::Vector3f(p.x() + wx, p.y(), p.z())); if (neighbor) neighbors.push_back(neighbor);
    neighbor = getLeaf(Eigen::Vector3f(p.x() - wx, p.y(), p.z())); if (neighbor) neighbors.push_back(neighbor);
    neighbor = getLeaf(Eigen::Vector3f(p.x(), p.y() + wy, p.z())); if (neighbor) neighbors.push_back(neighbor);
    neighbor = getLeaf(Eigen::Vector3f(p.x(), p.y() - wy, p.z())); if (neighbor) neighbors.push_back(neighbor);
    neighbor = getLeaf(Eigen::Vector3f(p.x(), p.y(), p.z() + wz)); if (neighbor) neighbors.push_back(neighbor);
    neighbor = getLeaf(Eigen::Vector3f(p.x(), p.y(), p.z() - wz)); if (neighbor) neighbors.push_back(neighbor);

    // Optional: Remove duplicates if the same leaf ptr appears multiple times?
    // Usually not necessary unless leaf size is very small or points are exactly on boundaries.
    // std::sort(neighbors.begin(), neighbors.end());
    // neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

    return neighbors.size();
}


template <typename PointT>
int VoxelGridCovariance<PointT>::getNeighborhoodAtPoint1(
    const PointT& reference_point, std::vector<LeafConstPtr>& neighbors) const
{
    neighbors.clear();
    LeafConstPtr leaf = getLeaf(reference_point); // Get only the leaf containing the point
    if (leaf) {
        neighbors.push_back(leaf);
        return 1;
    }
    return 0;
}


} // namespace svn_ndt

#endif // SVN_NDT_VOXEL_GRID_COVARIANCE_IMPL_HPP_