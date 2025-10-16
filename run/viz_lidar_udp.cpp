#include <pipeline.hpp>

static std::atomic<bool> running = true;

void signal_handler(int) {
    running = false;
}

int main() {

    struct sigaction sa = {};
    sa.sa_handler = signal_handler;
    sigaction(SIGINT, &sa, nullptr);

    std::string meta_path = "../config/lidar_meta_berlin.json";
    std::string param_path = "../config/lidar_config_berlin.json";
    LidarCallback callback(meta_path, param_path);

    FrameQueue<LidarFrame> lidarQueue;
    FrameQueue<DataBuffer> packetQueue;
    
    // Using a shared_ptr for frameid allows safe sharing with the lambda
    auto last_frame_id = std::make_shared<uint16_t>(0);

    boost::asio::io_context io_context;

    UdpSocketConfig lidarUdpConfig;
    lidarUdpConfig.host = "192.168.75.10";
    lidarUdpConfig.multicastGroup = std::nullopt;
    lidarUdpConfig.localInterfaceIp = "192.168.75.10";
    lidarUdpConfig.port = 7502;
    lidarUdpConfig.bufferSize = 24832;
    lidarUdpConfig.receiveTimeout = std::chrono::milliseconds(10000); 
    lidarUdpConfig.reuseAddress = true; 
    lidarUdpConfig.enableBroadcast = false; 
    lidarUdpConfig.ttl =  std::nullopt; 

    auto data_callback = [&packetQueue](std::unique_ptr<DataBuffer> packet_ptr) {
        if (!running) return;
        packetQueue.push(std::move(packet_ptr));
    };

    auto lidar_thread = std::thread([&callback, &packetQueue, &lidarQueue, last_frame_id]() {
        try {
            while (running) {
                auto packet_ptr = packetQueue.pop();
                if (!packet_ptr) {
                    break;
                }
                auto frame = std::make_unique<LidarFrame>();
                callback.DecodePacketRng19(*packet_ptr, *frame);
                if (frame->numberpoints > 0 && frame->frame_id != *last_frame_id) {
                    *last_frame_id = frame->frame_id;
                    std::cout << "Processed complete frame " << frame->frame_id 
                            << " with " << frame->numberpoints << " points\n";
                    lidarQueue.push(std::move(frame));
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "lidar thread error: " << e.what() << "\n";
        }
    });

    auto error_callback = [](const boost::system::error_code& ec) {
       if (running) {
            std::cerr << "LiDAR IO error: " << ec.message() << " (code: " << ec.value() << ")\n";
        }
    };

    auto socket = UdpSocket::create(io_context, lidarUdpConfig, data_callback, error_callback);

    auto io_thread = std::thread([&io_context]() {
        try {
                io_context.run();
        } catch (const std::exception& e) {
            std::cerr << "IO context error: " << e.what() << "\n";
        }
    });

    // In viz_lidar_udp.cpp
    auto viz_thread = std::thread([&lidarQueue]() {
        auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("LiDAR Visualizer");
        viewer->setBackgroundColor(0.1, 0.1, 0.1);
        viewer->addCoordinateSystem(10.0, "coord"); //
        viewer->initCameraParameters(); //

        const Eigen::Vector3d kCameraOffset(0.0, 0.0, -250.0);
        const Eigen::Vector3d kUpVector(1.0, 0.0, 0.0);
        Eigen::Vector3d target_focal_point(0.0, 0.0, 0.0);

        viewer->setCameraPosition(
                kCameraOffset.x(), kCameraOffset.y(), kCameraOffset.z(),
                target_focal_point.x(), target_focal_point.y(), target_focal_point.z(),
                kUpVector.x(), kUpVector.y(), kUpVector.z()
            );

        while (running && !viewer->wasStopped()) {
            auto frame_ptr = lidarQueue.pop(); //
            if (!frame_ptr) break;

            // Create cloud in original NED coordinate system
            PCLPointCloud structcloud = frame_ptr->toPCLPointCloud(); //
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = structcloud.pointsBody.makeShared();
            cloud->header.stamp = static_cast<std::uint64_t>(frame_ptr->timestamp_end * 1e9); //
            
            // --- Visualization ---
            pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color_handler(cloud, "intensity"); //
            if (!viewer->updatePointCloud(cloud, color_handler, "lidar_cloud")) { //
                viewer->addPointCloud(cloud, color_handler, "lidar_cloud"); //
            }
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "lidar_cloud"); //
            
            viewer->spinOnce(1); //
        }
    });

    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    socket->stop();
    packetQueue.stop();
    lidarQueue.stop();
    io_context.stop();
    if (lidar_thread.joinable()) lidar_thread.join();
    if (viz_thread.joinable()) viz_thread.join();
    if (io_thread.joinable()) io_thread.join();

    return 0;
}