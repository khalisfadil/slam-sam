// viz_lidar_udp.cpp
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <chrono>
#include <cstdint>  // Added for std::uint64_t
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <boost/asio.hpp>

#include "lidarcallback.hpp"  // Includes LidarFrame and LidarCallback
#include "udpsocket.hpp"      // UDP socket for packet reception

// Thread-safe queue for PCL clouds
class CloudQueue {
public:
    void push(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(cloud);
        cv_.notify_one();
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || stopped_; });
        if (queue_.empty() && stopped_) return nullptr;
        auto cloud = queue_.front();
        queue_.pop();
        return cloud;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cv_.notify_all();
    }

private:
    std::queue<pcl::PointCloud<pcl::PointXYZI>::Ptr> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stopped_ = false;
};

int main() {
    // Initialize LidarCallback (replace with your JSON file paths)
    std::string meta_path = "../config/lidar_meta_berlin.json";
    std::string param_path = "../config/lidar_config_berlin.json";
    LidarCallback callback(meta_path, param_path);

    CloudQueue cloud_queue;
    LidarFrame frame;

    // Boost.Asio context for UDP
    boost::asio::io_context io_context;

    // UDP socket configuration (adjust for your LiDAR, e.g., Ouster OS1)
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

    // Data callback to process incoming packets
    auto data_callback = [&callback, &cloud_queue, &frame](const DataBuffer& packet) {
        callback.DecodePacketRng19(packet, frame);
        if (frame.numberpoints > 0) {
            auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>(frame.toPCLPointCloud());
            cloud->header.stamp = static_cast<std::uint64_t>(frame.timestamp * 1e9);  // Fixed: pcl::uint64_t -> std::uint64_t
            cloud_queue.push(cloud);
            std::cout << "Decoded frame " << frame.frame_id << " with " << frame.numberpoints << " points\n";
        }
    };

    // Error callback for UDP errors
    auto error_callback = [](const boost::system::error_code& ec) {
        std::cerr << "UDP error: " << ec.message() << "\n";
    };

    // Create UDP socket
    auto socket = UdpSocket::create(io_context, config, data_callback, error_callback);

    // Visualization thread
    auto viz_thread = std::thread([&cloud_queue]() {
        auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("LiDAR Visualizer");
        viewer->setBackgroundColor(0, 0, 0);
        viewer->addCoordinateSystem(1.0, "coord");
        viewer->initCameraParameters();

        while (!viewer->wasStopped()) {
            auto cloud = cloud_queue.pop();
            if (!cloud) break;  // Stopped signal
            if (cloud->size() > 50000) {
                pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZI>);
                pcl::VoxelGrid<pcl::PointXYZI> vg;
                vg.setLeafSize(0.1f, 0.1f, 0.1f);
                vg.setInputCloud(cloud);
                vg.filter(*downsampled);
                cloud = downsampled;
            }

            pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color_handler(cloud, "intensity");
            if (!viewer->updatePointCloud(cloud, color_handler, "lidar_cloud")) {
                viewer->addPointCloud(cloud, color_handler, "lidar_cloud");
            }
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "lidar_cloud");
            viewer->spinOnce(16);  // ~60 FPS
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
    });

    // Run IO context in a separate thread
    auto io_thread = std::thread([&io_context]() {
        try {
            io_context.run();
        } catch (const std::exception& e) {
            std::cerr << "IO context error: " << e.what() << "\n";
        }
    });

    // Wait for user input to stop
    std::cout << "Press Enter to stop...\n";
    std::cin.get();

    // Cleanup
    socket->stop();
    cloud_queue.stop();
    io_context.stop();
    if (viz_thread.joinable()) viz_thread.join();
    if (io_thread.joinable()) io_thread.join();

    return 0;
}