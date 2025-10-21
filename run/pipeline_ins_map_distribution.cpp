#include <pipeline.hpp>

// Use a single atomic for running, and a CV to notify main
static std::atomic<bool> running = true;
static std::mutex running_mutex;
static std::condition_variable running_cv;

void signal_handler(int) {
    std::cout << "\nCaught signal, shutting down..." << std::endl;
    {
        std::lock_guard<std::mutex> lock(running_mutex);
        running = false;
    }
    running_cv.notify_one(); // Notify main thread to exit
}

int main() {
    // ##########################################################################
    // Signal Handling
    // ##########################################################################
    struct sigaction sa = {};
    sa.sa_handler = signal_handler;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    // ##########################################################################
    // Config and Callback Setup
    // ##########################################################################
    std::string meta_lidar = "../config/lidar_meta_berlin.json";
    std::string param_lidar = "../config/lidar_config_berlin.json";
    std::string param_imu = "../config/imu_config_berlin.json";
    std::string param_reg = "../config/register_config.json";
    
    LidarCallback lidarCallback(meta_lidar, param_lidar);
    CompCallback compCallback(param_imu);
    RegisterCallback registerCallback(param_reg);   

    // ##########################################################################
    // Queues and Object Pools
    // ##########################################################################
    FrameQueue<LidarFrame> frameLidarQueue;
    FrameQueue<DataBuffer> packetLidarQueue;
    FrameQueue<std::deque<CompFrame>> frameCompQueue;
    FrameQueue<DataBuffer> packetCompQueue;
    FrameQueue<FrameData> frameDataQueue;

    ObjectPool<std::deque<CompFrame>> comp_window_pool(8);
    ObjectPool<FrameData> frame_data_pool(8);

    // ##########################################################################
    // Shared State
    // ##########################################################################
    auto latest_frame_id = std::make_shared<uint16_t>(0);
    auto comp_window_size = std::make_shared<size_t>(24);
    auto comp_latest_timestamp = std::make_shared<double>(0);
    auto lidar_keyframe_timestamp = std::make_shared<double>(0);

    auto comp_window = std::make_shared<std::deque<CompFrame>>();

    // ##########################################################################
    // Compass UDP Socket (INS)
    // ##########################################################################
    boost::asio::io_context comp_iocontext;
    UdpSocketConfig compUdpConfig;
    compUdpConfig.host = "192.168.75.10";
    compUdpConfig.multicastGroup = std::nullopt;
    compUdpConfig.localInterfaceIp = "192.168.75.10";
    compUdpConfig.port = 6597;
    compUdpConfig.bufferSize = 105;
    compUdpConfig.receiveTimeout = std::chrono::milliseconds(10000); 
    compUdpConfig.reuseAddress = true; 
    compUdpConfig.enableBroadcast = false; 
    compUdpConfig.ttl =  std::nullopt; 
    
    auto comp_callback = [&packetCompQueue](std::unique_ptr<DataBuffer> packet_ptr) {
        if (!running) return;
        packetCompQueue.push(std::move(packet_ptr));
    };
    auto comp_error_callback = [](const boost::system::error_code& ec) {
        if (running) {
            std::cerr << "Compass IO error: " << ec.message() << "\n";
        }
    };
    auto comp_socket = UdpSocket::create(comp_iocontext, compUdpConfig, comp_callback, comp_error_callback);
    auto comp_iothread = std::thread([&comp_iocontext]() {
        try {
            comp_iocontext.run();
        } catch (const std::exception& e) {
            std::cerr << "Comp IO context error: " << e.what() << "\n";
        }
    });

    // ##########################################################################
    // Lidar UDP Socket
    // ##########################################################################
    boost::asio::io_context lidar_iocontext;
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

    auto lidar_callback = [&packetLidarQueue](std::unique_ptr<DataBuffer> packet_ptr) {
        if (!running) return;
        packetLidarQueue.push(std::move(packet_ptr));
    };
    auto lidar_error_callback = [](const boost::system::error_code& ec) {
       if (running) {
            std::cerr << "LiDAR IO error: " << ec.message() << "\n";
        }
    };
    auto lidar_socket = UdpSocket::create(lidar_iocontext, lidarUdpConfig, lidar_callback, lidar_error_callback);
    auto lidar_iothread = std::thread([&lidar_iocontext]() {
        try {
            lidar_iocontext.run();
        } catch (const std::exception& e) {
            std::cerr << "Lidar IO context error: " << e.what() << "\n";
        }
    });

    // ##########################################################################
    // Lidar Processing Thread
    // ##########################################################################
    auto lidar_processing_thread = std::thread([&lidarCallback, &packetLidarQueue, &frameLidarQueue]() {
        while (running) {
            auto packet_ptr = packetLidarQueue.pop();
            if (!packet_ptr) {
                break; // Queue was stopped
            }
            auto frame_ptr = lidarCallback.DecodePacket(*packet_ptr);
            if (frame_ptr) {
                std::cout << "Decoded frame " << frame_ptr->frame_id << " with " << frame_ptr->numberpoints << " points\n";
                frameLidarQueue.push(std::move(frame_ptr));
            }
        }
        std::cout << "Lidar processing thread stopped.\n";
    });

    // ##########################################################################
    // Compass (INS) Processing Thread (Refactored)
    // ##########################################################################
    auto comp_processing_thread = std::thread([
            &compCallback, &packetCompQueue, &frameCompQueue, 
            &comp_window_pool, 
            comp_latest_timestamp, comp_window, comp_window_size
        ]() {
        while (running) {
            auto packet_ptr = packetCompQueue.pop();
            if (!packet_ptr) {
                break; // Queue was stopped
            }
            auto frame_ptr = compCallback.DecodePacket(*packet_ptr);
            if (!frame_ptr) {
                continue; // Decode failed, loop again
            }
            if (frame_ptr->timestamp_20 == *comp_latest_timestamp) {
                compCallback.ReturnFrameToPool(std::move(frame_ptr)); // Return and skip
                continue;
            }
            *comp_latest_timestamp = frame_ptr->timestamp_20;
            comp_window->push_back(std::move(*frame_ptr));
            compCallback.ReturnFrameToPool(std::move(frame_ptr));
            if (comp_window->size() > *comp_window_size) {
                comp_window->pop_front();
            }
            if (comp_window->size() == *comp_window_size) {
                auto comp_window_copy_ptr = comp_window_pool.Get();
                *comp_window_copy_ptr = *comp_window; 
                frameCompQueue.push(std::move(comp_window_copy_ptr));
            }
        }
        std::cout << "Compass processing thread stopped.\n";
    });

    // ##########################################################################
    // Sync Processing Thread
    // ##########################################################################
    auto sync_thread = std::thread([&lidarCallback, &compCallback, &frameLidarQueue, &frameCompQueue, &frameDataQueue, 
                                &comp_window_pool, &frame_data_pool, // <-- Capture the pool
                                lidar_keyframe_timestamp]() {

        auto getInterpolated = [&](double target_time, const std::unique_ptr<std::deque<CompFrame>>& window) -> CompFrame {
            if (!window || window->empty()) {
                return CompFrame(); // Return a default/empty frame
            }
            if (target_time <= window->front().timestamp_20) return window->front();
            if (target_time >= window->back().timestamp_20) return window->back();
            for (size_t i = 0; i < window->size() - 1; ++i) {
                const CompFrame& a = (*window)[i];
                const CompFrame& b = (*window)[i + 1];
                if (a.timestamp_20 <= target_time && target_time <= b.timestamp_20) {
                    double t = (b.timestamp_20 - a.timestamp_20 > 1e-9) ? (target_time - a.timestamp_20) / (b.timestamp_20 - a.timestamp_20) : 0.0;
                    return a.linearInterpolate(a, b, t);
                }
            }
            return window->back();
        };
        std::unique_ptr<std::deque<CompFrame>> comp_window_frame_ptr = nullptr;
        bool is_first_frame = true;
        while (running) {  
            auto lidar_frame_ptr = frameLidarQueue.pop();
            if (!lidar_frame_ptr) {
                break; // Queue was stopped //will break the outer while loop
            }
            if (lidar_frame_ptr->timestamp_points.size() < 2) { 
                std::cerr << "LiDAR frame " << lidar_frame_ptr->frame_id << " has insufficient points, skipping.\n";
                lidarCallback.ReturnFrameToPool(std::move(lidar_frame_ptr));
                continue;
            }

            const double max_lidar_time = lidar_frame_ptr->timestamp_points.back();

            if (is_first_frame) { 
                *lidar_keyframe_timestamp = max_lidar_time; 
                is_first_frame = false; 
                lidarCallback.ReturnFrameToPool(std::move(lidar_frame_ptr));
                continue; 
            }

            const double& start_interval = *lidar_keyframe_timestamp; 
            const double& end_interval = max_lidar_time;

            bool data_gap_detected = false;
            
            while (running) {
                comp_window_frame_ptr = frameCompQueue.pop();
                if (!comp_window_frame_ptr) {
                    break;  // break inner while loop
                }
                if (comp_window_frame_ptr->back().timestamp_20 < end_interval) { 
                    comp_window_pool.Return(std::move(comp_window_frame_ptr)); // <-- Return before getting a new one
                    continue; 
                }
                if (comp_window_frame_ptr->front().timestamp_20 > start_interval) { //
                    std::cerr << std::setprecision(12) << "CRITICAL: Data gap detected in compass stream. "
                            << "Required interval starts at " << start_interval
                            << " but available data starts at " << comp_window_frame_ptr->front().timestamp_20 << ".\n";
                    data_gap_detected = true; //
                    comp_window_pool.Return(std::move(comp_window_frame_ptr));
                    break; // break inner while loop, data grap detected break
                }
                break; // break inner while loop, success break
            }
            if (!running || !comp_window_frame_ptr) {
                lidarCallback.ReturnFrameToPool(std::move(lidar_frame_ptr));
                break;
            }
            if (data_gap_detected) { 
                std::cerr << "Skipping LiDAR frame " << lidar_frame_ptr->frame_id << " due to compass data gap.\n";
                lidarCallback.ReturnFrameToPool(std::move(lidar_frame_ptr)); // <-- Add this
                continue; 
            }
            *lidar_keyframe_timestamp = end_interval;
            auto data_frame_ptr = frame_data_pool.Get();
            data_frame_ptr->points = lidar_frame_ptr->toPCLPointCloud(); 
            data_frame_ptr->timestamp = end_interval;
            data_frame_ptr->ins.reserve(comp_window_frame_ptr->size());
            data_frame_ptr->ins.push_back(getInterpolated(start_interval, comp_window_frame_ptr));
            for (const auto& data : *comp_window_frame_ptr) {
                if (data.timestamp_20 > start_interval && data.timestamp_20 < end_interval) { 
                    data_frame_ptr->ins.push_back(data);
                }
            }
            data_frame_ptr->ins.push_back(getInterpolated(end_interval, comp_window_frame_ptr));
            frameDataQueue.push(std::move(data_frame_ptr));
            lidarCallback.ReturnFrameToPool(std::move(lidar_frame_ptr));
            comp_window_pool.Return(std::move(comp_window_frame_ptr));
            
        }
        std::cout << "Sync processing thread stopped.\n";                          
    });

    // ##########################################################################
    // Cummulation Thread
    // ##########################################################################
    auto cumm_thread = std::thread([&registerCallback, &frameDataQueue, &frame_data_pool]() {

        // =================================================================================
        // STATE & ARCHIVE VARIABLES
        // =================================================================================
        PointsHashMap pointsArchive;
        Eigen::Vector3d ins_rlla = Eigen::Vector3d::Zero();
        bool is_first_keyframe = true;

        // =================================================================================
        // NDT SETUP
        // =================================================================================
        pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>::Ptr ndt_omp = nullptr;
        if (registerCallback.registration_method_ == "NDT") {
            ndt_omp.reset(new pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>());
            ndt_omp->setNumThreads(registerCallback.num_threads_);
            ndt_omp->setResolution(registerCallback.ndt_resolution_);
            ndt_omp->setTransformationEpsilon(registerCallback.ndt_transform_epsilon_);
            ndt_omp->setRegularizationScaleFactor(registerCallback.regularization_scale_factor_);

            if (registerCallback.ndt_neighborhood_search_method_ == "DIRECT1") {
                ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT1);
            } else if (registerCallback.ndt_neighborhood_search_method_ == "DIRECT7") {
                ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT7);
            } else if (registerCallback.ndt_neighborhood_search_method_ == "KDTREE") {
                ndt_omp->setNeighborhoodSearchMethod(pclomp::KDTREE);
            } else {
                std::cout << "Warning: Invalid NDT search method. Defaulting to DIRECT7." << std::endl;
                ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT7);
            }
        } 
        
        while (running){
            auto data_frame_ptr = frameDataQueue.pop();
            if (!data_frame_ptr) {
                break;
            }
            pcl::PointCloud<pcl::PointXYZI>::Ptr pointsBody(new pcl::PointCloud<pcl::PointXYZI>());
            *pointsBody = std::move(data_frame_ptr->points.pointsBody);
            const auto& ins = data_frame_ptr->ins.back();
            const uint64_t& id = data_frame_ptr->points.frame_id;
            const auto& timestamp = data_frame_ptr->timestamp;
            frame_data_pool.Return(std::move(data_frame_ptr));
            const Eigen::Vector3d ins_lla{ins.latitude_20, ins.longitude_20, ins.altitude_20};
            Eigen::Quaternionf ins_quat{ins.qw_20, ins.qx_20, ins.qy_20, ins.qz_20};
            const Eigen::Matrix3d Cb2m = ins_quat.toRotationMatrix().cast<double>();
            const gtsam::Rot3 ins_Cb2m{Cb2m};
            const gtsam::Vector3 ins_vNED{ins.velocityNorth_20, ins.velocityEast_20, ins.velocityDown_20};

            if (is_first_keyframe) {
                ins_rlla = ins_lla;
                is_first_keyframe = false;
            } 

            const gtsam::Point3 ins_tb2m{registerCallback.lla2ned(ins_lla.x(), ins_lla.y(), ins_lla.z(), ins_rlla.x(), ins_rlla.y(), ins_rlla.z())};
            gtsam::NavState current_ins_state = gtsam::NavState{ins_Cb2m, ins_tb2m, ins_vNED};
            pcl::PointCloud<pcl::PointXYZI>::Ptr pointsMap(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::transformPointCloud(*pointsBody, *pointsMap, current_ins_state.pose().matrix());
            pointsArchive[id] = {std::move(pointsMap), timestamp};
        }
        // This part runs *after* the while(running) loop has exited (i.e., on shutdown)
        std::cout << "Accumulating final map from " << pointsArchive.size() << " keyframes..." << std::endl;

        pcl::PointCloud<pcl::PointXYZI>::Ptr map(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr ds_map(new pcl::PointCloud<pcl::PointXYZI>());
        
        pcl::VoxelGrid<pcl::PointXYZI> vg;
        auto vs = registerCallback.mapvoxelsize_;
        vg.setLeafSize(vs, vs, vs); // Set leaf size as float

        // Correctly iterate the tsl::robin_map
        for (const auto& pair : pointsArchive) {
            if (pair.second.points) {
                *map += *(pair.second.points);
            }
        }

        if (map->empty()) {
            std::cout << "Map is empty, nothing to filter or export." << std::endl;
        } else {
            std::cout << "Map accumulated with " << map->size() << " points. Downsampling..." << std::endl;
            vg.setInputCloud(map);
            vg.filter(*ds_map);
            std::cout << "Map downsampled to " << ds_map->size() << " points." << std::endl;

            // --- FIX: Only run NDT export if NDT was configured ---
            if (registerCallback.registration_method_ == "NDT" && ndt_omp != nullptr) {
                std::cout << "Exporting NDT data..." << std::endl;
                ndt_omp->setInputTarget(ds_map); // Now safe to call
                
                // Explicitly state the template type <pcl::PointXYZI>
                auto exported_data = extractNdtData<pcl::PointXYZI>(ndt_omp, ds_map);
                
                writeNdtDataToFiles(
                    exported_data,
                    "../output/ndt_ellipsoids.txt",
                    "../output/ndt_voxels.txt",
                    "../output/map_points.txt"
                );
            } else {
                 std::cout << "Skipping NDT data export (method was not NDT or NDT object is null)." << std::endl;
                 // Optionally save just the downsampled map if NDT wasn't used
                 // pcl::io::savePCDFileASCII("../output/downsampled_map.pcd", *ds_map);
            }
            // --- End Fix ---
        }
        std::cout << "Cummulation thread finished." << std::endl; // Added for clarity
    }); // End of cumm_thread lambda

    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    lidar_socket->stop();
    comp_socket->stop();

    // comp_iocontext.stop();
    // lidar_iocontext.stop();
    packetCompQueue.stop();
    packetLidarQueue.stop();
    frameLidarQueue.stop();
    frameCompQueue.stop();
    frameDataQueue.stop();

    std::cout << "Joining comp_iothread..." << std::endl;
    if (comp_iothread.joinable()) comp_iothread.join();
    std::cout << "Joined comp_iothread." << std::endl;

    std::cout << "Joining lidar_iothread..." << std::endl;
    if (lidar_iothread.joinable()) lidar_iothread.join();
    std::cout << "Joined lidar_iothread." << std::endl;

    std::cout << "Joining lidar_processing_thread..." << std::endl;
    if (lidar_processing_thread.joinable()) lidar_processing_thread.join();
    std::cout << "Joined lidar_processing_thread." << std::endl;

    std::cout << "Joining comp_processing_thread..." << std::endl;
    if (comp_processing_thread.joinable()) comp_processing_thread.join();
    std::cout << "Joining comp_processing_thread..." << std::endl;

    std::cout << "Joining sync_thread..." << std::endl;
    if (sync_thread.joinable()) sync_thread.join();
    std::cout << "Joining sync_thread..." << std::endl;

    std::cout << "Joining cumm_thread..." << std::endl;
    if (cumm_thread.joinable()) cumm_thread.join();
    std::cout << "Joining cumm_thread..." << std::endl;

    std::cout << "All threads joined. Returning from main..." << std::endl;
    
    return 0;
}