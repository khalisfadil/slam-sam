// viz_lidar_udp.cpp (Optimized)
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <boost/asio.hpp>

#include <lidarcallback.hpp>
#include <udpsocket.hpp>

// OPTIMIZATION 1: The queue now holds the lightweight LidarFrame struct.
// Using std::unique_ptr to avoid copying large frame objects.
using LidarFramePtr = std::unique_ptr<LidarFrame>;

class LidarFrameQueue {
public:
    void push(LidarFramePtr frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(frame));
        cv_.notify_one();
    }

    LidarFramePtr pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || stopped_; });
        if (queue_.empty() && stopped_) return nullptr;
        LidarFramePtr frame = std::move(queue_.front());
        queue_.pop();
        return frame;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cv_.notify_all();
    }

private:
    std::queue<LidarFramePtr> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stopped_ = false;
};

int main() {
    std::string meta_path = "../config/lidar_meta_berlin.json";
    std::string param_path = "../config/lidar_config_berlin.json";
    LidarCallback callback(meta_path, param_path);

    LidarFrameQueue frame_queue; // Use the new queue
    
    // Using a shared_ptr for frameid allows safe sharing with the lambda
    auto last_frame_id = std::make_shared<uint16_t>(0);

    boost::asio::io_context io_context;

    UdpSocketConfig config;
    config.host = "192.168.75.10";
    config.multicastGroup = std::nullopt;
    config.localInterfaceIp = "192.168.75.10";
    config.port = 7502;
    config.bufferSize = 24832;
    config.receiveTimeout = std::chrono::milliseconds(10000); 
    config.reuseAddress = true; 
    config.enableBroadcast = false; 
    config.ttl =  std::nullopt; 

    // OPTIMIZATION 1: data_callback is now very lightweight.
    auto data_callback = [&callback, &frame_queue, last_frame_id](const DataBuffer& packet) {
        // Use a unique_ptr to manage the frame's lifetime.
        auto frame = std::make_unique<LidarFrame>();
        callback.DecodePacketRng19(packet, *frame);
        
        if (frame->numberpoints > 0 && frame->frame_id != *last_frame_id) {
            *last_frame_id = frame->frame_id;
            std::cout << "Decoded frame " << frame->frame_id << " with " << frame->numberpoints << " points\n";
            // Move the frame pointer into the queue. No heavy copying.
            frame_queue.push(std::move(frame));
        }
    };

    auto error_callback = [](const boost::system::error_code& ec) {
        std::cerr << "UDP error: " << ec.message() << "\n";
    };

    auto socket = UdpSocket::create(io_context, config, data_callback, error_callback);

    // In viz_lidar_udp.cpp

    auto viz_thread = std::thread([&frame_queue]() {
        auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("LiDAR Visualizer");
        viewer->setBackgroundColor(0, 0, 0); //
        viewer->addCoordinateSystem(10.0, "coord"); //
        viewer->initCameraParameters(); //

        // --- Filter setup ---
        // Intensity filter
        // pcl::PassThrough<pcl::PointXYZI> pass_intensity; //
        // pass_intensity.setFilterFieldName("intensity"); //
        // pass_intensity.setFilterLimits(0.0f, 100.0f); //

        // Spatial filter
        pcl::PassThrough<pcl::PointXYZI> pass_spatial; //
        // NOTE: Changed filter field to "y", which is now the forward (North) axis
        pass_spatial.setFilterFieldName("z");
        pass_spatial.setFilterLimits(-200.0, 5.0); // Adjust limits for forward/backward
        
        // VoxelGrid filter
        pcl::VoxelGrid<pcl::PointXYZI> vg; //
        vg.setLeafSize(0.5f, 0.5f, 0.5f); //
        
        // Point cloud pointers for each stage
        pcl::PointCloud<pcl::PointXYZI>::Ptr intensity_filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>()); //
        pcl::PointCloud<pcl::PointXYZI>::Ptr spatial_filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>()); //
        pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZI>()); //

        while (!viewer->wasStopped()) {
            auto frame_ptr = frame_queue.pop(); //
            if (!frame_ptr) break;

            // Create cloud in original NED coordinate system
            pcl::PointCloud<pcl::PointXYZI> cloud = frame_ptr->toPCLPointCloud(); //
            cloud.header.stamp = static_cast<std::uint64_t>(frame_ptr->timestamp * 1e9); //
            
            // --- NEW: Add transformation loop ---
            // Apply the (x, y, z) -> (y, x, -z) transform to match the PCL viewer's frame.
            for (auto& point : cloud.points) {
                float original_x = point.x;
                point.x = point.y;    // New X is Old Y (East -> PCL Right)
                point.y = -original_x; // New Y is Old X (North -> PCL Up)
                point.z = point.z;   // New Z is -Old Z (Down -> PCL Out of screen)
            }
            // --- End of new code ---
            
            // --- Filtering chain ---
            // pass_intensity.setInputCloud(cloud.makeShared()); //
            // pass_intensity.filter(*intensity_filtered_cloud); //

            pass_spatial.setInputCloud(cloud.makeShared()); //
            pass_spatial.filter(*spatial_filtered_cloud); //

            vg.setInputCloud(spatial_filtered_cloud); //
            vg.filter(*downsampled_cloud); //
            
            // --- Visualization ---
            pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color_handler(downsampled_cloud, "intensity"); //
            if (!viewer->updatePointCloud(downsampled_cloud, color_handler, "lidar_cloud")) { //
                viewer->addPointCloud(downsampled_cloud, color_handler, "lidar_cloud"); //
            }
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "lidar_cloud"); //
            
            viewer->spinOnce(1); //
        }
    });

    auto io_thread = std::thread([&io_context]() {
        try {
            io_context.run();
        } catch (const std::exception& e) {
            std::cerr << "IO context error: " << e.what() << "\n";
        }
    });

    std::cout << "Press Enter to stop...\n";
    std::cin.get();

    socket->stop();
    frame_queue.stop();
    io_context.stop();
    if (viz_thread.joinable()) viz_thread.join();
    if (io_thread.joinable()) io_thread.join();

    return 0;
}