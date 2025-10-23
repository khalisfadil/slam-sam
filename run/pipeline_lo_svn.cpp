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

struct LoSvnVisData {
    std::shared_ptr<PointsHashMap> points;
    std::shared_ptr<PoseHashMap> insposes;
    std::shared_ptr<PoseHashMap> loposes;
};

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
    FrameQueue<LoSvnVisData> visDataQueue;
    ObjectPool<std::deque<CompFrame>> comp_window_pool(8);
    ObjectPool<FrameData> frame_data_pool(8);
    ObjectPool<LoSvnVisData> vis_data_pool(8);

    // ##########################################################################
    // Shared State
    // ##########################################################################
    auto lidar_latest_frame_id = std::make_shared<uint16_t>(std::numeric_limits<uint16_t>::min());
    auto lidar_keyframe_timestamp = std::make_shared<double>(std::numeric_limits<double>::lowest());
    auto comp_latest_timestamp = std::make_shared<double>(std::numeric_limits<double>::lowest());
    auto comp_window_size = std::make_shared<size_t>(24);
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
    auto lidar_processing_thread = std::thread([&lidarCallback, &packetLidarQueue, &frameLidarQueue,
                                                lidar_latest_frame_id]() {
        while (running) {
            auto packet_ptr = packetLidarQueue.pop();
            if (!packet_ptr) {
                break; // Queue was stopped
            }
            auto frame_ptr = lidarCallback.DecodePacket(*packet_ptr);
            if (!frame_ptr) {
                continue; // Decode failed, loop again
            }
            if (frame_ptr->frame_id <= *lidar_latest_frame_id) {
                lidarCallback.ReturnFrameToPool(std::move(frame_ptr)); // Return and skip
                continue;
            }
            *lidar_latest_frame_id = frame_ptr->frame_id;
            std::cout << "Decoded frame " << frame_ptr->frame_id << " with " << frame_ptr->numberpoints << " points\n";
            frameLidarQueue.push(std::move(frame_ptr));
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
        // std::cout << "Compass window pool size: " << comp_window_pool.GetAvailableCount() << "\n" << std::endl;   
        std::cout << "Sync processing thread stopped.\n";                          
    });

    // ##########################################################################
    // Gtsam Processing Thread
    // ##########################################################################
    auto lo_thread = std::thread([&registerCallback, &compCallback, &frameDataQueue, &visDataQueue,
                                    &frame_data_pool, &vis_data_pool]() {
        // =================================================================================
        // A. SVN NDT SETUP
        // =================================================================================
        std::unique_ptr<svn_ndt::SvnNormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>> svn_ndt_ptr = nullptr;
        if (registerCallback.registration_method_ == "SVNNDT") {
            svn_ndt_ptr = std::make_unique<svn_ndt::SvnNormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>>();
            svn_ndt_ptr->setResolution(registerCallback.svn_ndt_resolution_);
            svn_ndt_ptr->setParticleCount(registerCallback.svn_ndt_number_particle_);
            svn_ndt_ptr->setMaxIterations(registerCallback.svn_ndt_max_iterations_);       // Max SVN loops
            svn_ndt_ptr->setKernelBandwidth(registerCallback.svn_ndt_kernel_bandwith_);    // h (needs tuning)
            svn_ndt_ptr->setStepSize(registerCallback.svn_ndt_step_size_);           // Epsilon
            svn_ndt_ptr->setEarlyStopThreshold(registerCallback.svn_ndt_stop_threshold_); // Convergence threshold
            svn_ndt_ptr->setOutlierRatio(registerCallback.svn_ndt_set_outlier_ratio_);

            if (registerCallback.svn_ndt_neighborhood_search_method_ == "DIRECT1") {
                svn_ndt_ptr->setNeighborhoodSearchMethod(svn_ndt::NeighborSearchMethod::DIRECT1);
            } else if (registerCallback.ndt_neighborhood_search_method_ == "DIRECT7") {
                svn_ndt_ptr->setNeighborhoodSearchMethod(svn_ndt::NeighborSearchMethod::DIRECT7);
            } else if (registerCallback.ndt_neighborhood_search_method_ == "KDTREE") {
                svn_ndt_ptr->setNeighborhoodSearchMethod(svn_ndt::NeighborSearchMethod::KDTREE);
            } else {
                std::cout << "Warning: Invalid SVN NDT search method. Defaulting to DIRECT7." << std::endl;
                svn_ndt_ptr->setNeighborhoodSearchMethod(svn_ndt::NeighborSearchMethod::DIRECT7);
            }
        }
        const int targetWinSize = 5;            // how many frame from the previous lidar odometry to use 
        std::deque<uint64_t> targetID;          // the ID

        // =================================================================================
        // STATE & ARCHIVE VARIABLES
        // =================================================================================
        Eigen::Vector3d ins_rlla = Eigen::Vector3d::Zero();
        bool is_first_keyframe = true;
        gtsam::Pose3 predTb2m;
        PointsHashMap pointsArchive;
        PoseHashMap insPoseArchive;
        PoseHashMap loPoseArchive;

        while (running) {
            auto data_frame_ptr = frameDataQueue.pop();
            if (!data_frame_ptr) {
                break;
            }
            pcl::PointCloud<pcl::PointXYZI>::Ptr pointsBody(new pcl::PointCloud<pcl::PointXYZI>());
            *pointsBody = std::move(data_frame_ptr->points.pointsBody);
            const auto& ins = data_frame_ptr->ins.back();
            const uint64_t& id = data_frame_ptr->points.frame_id;
            const auto& timestamp = data_frame_ptr->timestamp;

            const Eigen::Vector3d ins_lla{ins.latitude_20, ins.longitude_20, ins.altitude_20};
            Eigen::Quaternionf ins_quat{ins.qw_20, ins.qx_20, ins.qy_20, ins.qz_20};
            const Eigen::Matrix3d Cb2m = ins_quat.toRotationMatrix().cast<double>();
            const gtsam::Rot3 ins_Cb2m{Cb2m};
            const gtsam::Vector3 ins_vNED{ins.velocityNorth_20, ins.velocityEast_20, ins.velocityDown_20};
            gtsam::NavState current_ins_state{};

            if (is_first_keyframe) {
                ins_rlla = ins_lla;
                const gtsam::Point3 ins_tb2m{registerCallback.lla2ned(ins_lla.x(), ins_lla.y(), ins_lla.z(), ins_rlla.x(), ins_rlla.y(), ins_rlla.z())};
                current_ins_state = gtsam::NavState{ins_Cb2m, ins_tb2m, ins_vNED};
                predTb2m = current_ins_state.pose();
                pcl::PointCloud<pcl::PointXYZI>::Ptr pointsMap(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::transformPointCloud(*pointsBody, *pointsMap, current_ins_state.pose().matrix());
                pointsArchive[id] = {pointsMap, timestamp};
                insPoseArchive[id] = {current_ins_state.pose().matrix(), timestamp};
                loPoseArchive[id] = {current_ins_state.pose().matrix(), timestamp};
                targetID.push_back(id);
                is_first_keyframe = false;
            } else {
                const gtsam::Point3 ins_tb2m{registerCallback.lla2ned(ins_lla.x(), ins_lla.y(), ins_lla.z(), ins_rlla.x(), ins_rlla.y(), ins_rlla.z())};
                current_ins_state = gtsam::NavState{ins_Cb2m, ins_tb2m, ins_vNED};
                pcl::PointCloud<pcl::PointXYZI>::Ptr lidarFactorPointsTarget(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::PointCloud<pcl::PointXYZI>::Ptr lidarFactorPointsTargetDS(new pcl::PointCloud<pcl::PointXYZI>());
                *lidarFactorPointsTarget = (pointsArchive.at(targetID.back())).points;
                // for (const int& currID : targetID) {
                //     const auto& currlidarFactorPointsArchive = pointsArchive.at(currID);
                //     *lidarFactorPointsTarget += *currlidarFactorPointsArchive.points;
                // }
                // pcl::VoxelGrid<pcl::PointXYZI> vg;
                // auto vs = registerCallback.mapvoxelsize_;
                // vg.setLeafSize(vs, vs, vs);
                // vg.setInputCloud(std::move(lidarFactorPointsTarget));
                // vg.filter(*lidarFactorPointsTargetDS);
                svn_ndt_ptr->setInputTarget(lidarFactorPointsTarget);
                // svn_ndt::SvnNdtResult result = svn_ndt_ptr->align(*pointsBody, predTb2m);
                // predTb2m = result.final_pose;
                pcl::PointCloud<pcl::PointXYZI>::Ptr pointsMap(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::transformPointCloud(*pointsBody, *pointsMap, current_ins_state.pose().matrix());
                pointsArchive[id] = {pointsMap, timestamp};
                insPoseArchive[id] = {current_ins_state.pose().matrix(), timestamp};
                loPoseArchive[id] = {current_ins_state.pose().matrix(), timestamp};
                targetID.push_back(id);
                if (targetID.size() > targetWinSize) {
                    targetID.pop_front();
                }
            }
            // if(!pointsArchive.empty() || !insPoseArchive.empty() || !loPoseArchive.empty()) {
            //     auto vis_data_ptr = vis_data_pool.Get();
            //     vis_data_ptr->points = std::make_shared<PointsHashMap>(pointsArchive);
            //     vis_data_ptr->insposes = std::make_shared<PoseHashMap>(insPoseArchive);
            //     vis_data_ptr->loposes = std::make_shared<PoseHashMap>(loPoseArchive);
            //     visDataQueue.push(std::move(vis_data_ptr));
            // }
            frame_data_pool.Return(std::move(data_frame_ptr));
        }
        // std::cout << "Frame data pool size: " << frame_data_pool.GetAvailableCount() << "\n" << std::endl; 
        std::cout << "LO thread exiting\n";
    });

    // ##########################################################################
    // Viz Processing Thread
    // ##########################################################################
    // auto viz_thread = std::thread([&visDataQueue, &vis_data_pool]() { // <-- Capture vis_data_pool
    //     auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("GTSAM Optimized Map");
    //     viewer->setBackgroundColor(0.1, 0.1, 0.1);
    //     viewer->addCoordinateSystem(10.0, "world_origin");
    //     viewer->initCameraParameters();
        
    //     // --- Settings for Visualization ---
    //     const size_t kNumCloudsToViz = 5; // <--- Set to 5
    //     const double kSmoothingFactor = 0.1;
    //     const Eigen::Vector3d kCameraOffset(0.0, 0.0, -250.0);
    //     const Eigen::Vector3d kUpVector(1.0, 0.0, 0.0);
    //     Eigen::Vector3d target_focal_point(0.0, 0.0, 0.0);
    //     Eigen::Vector3d current_focal_point = target_focal_point;
    //     Eigen::Vector3d current_cam_pos = target_focal_point + kCameraOffset;
        
    //     // Voxel grid for downsampling visualization clouds
    //     pcl::VoxelGrid<pcl::PointXYZI> vg;
    //     vg.setLeafSize(0.5f, 0.5f, 0.5f); 
        
    //     // Trajectory clouds
    //     pcl::PointCloud<pcl::PointXYZRGB>::Ptr lo_trajectory_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    //     pcl::PointCloud<pcl::PointXYZRGB>::Ptr ins_trajectory_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    //     while (running && !viewer->wasStopped()) {
    //         auto vis_data_ptr = visDataQueue.pop();
    //         if (!vis_data_ptr) {
    //             break; // Queue was stopped
    //         }

    //         // --- 1. Update Camera Target ---
    //         if (vis_data_ptr->loposes && !vis_data_ptr->loposes->empty()) {
    //             auto max_it = std::max_element(
    //                 vis_data_ptr->loposes->begin(), 
    //                 vis_data_ptr->loposes->end(),
    //                 [](const auto& a, const auto& b) {
    //                     return a.first < b.first; // Compare keys
    //                 }
    //             );
    //             const auto& last_pose_matrix = max_it->second.pose;
    //             target_focal_point(0) = last_pose_matrix(0, 3);
    //             target_focal_point(1) = last_pose_matrix(1, 3);
    //             target_focal_point(2) = last_pose_matrix(2, 3);
    //         }

    //         // --- 2. Update Point Clouds (Last kNumCloudsToViz) ---
            
    //         // Remove all clouds from the previous iteration
    //         for (size_t i = 0; i < kNumCloudsToViz; ++i) {
    //             viewer->removePointCloud("point_cloud_" + std::to_string(i));
    //         }

    //         if (vis_data_ptr->points && !vis_data_ptr->points->empty()) {
                
    //             // 1. Get all keys and sort them (oldest to newest)
    //             std::vector<uint64_t> keys;
    //             keys.reserve(vis_data_ptr->points->size());
    //             for (const auto& pair : *(vis_data_ptr->points)) {
    //                 keys.push_back(pair.first);
    //             }
    //             std::sort(keys.begin(), keys.end());

    //             // 2. Find the index to start from to get the last 5 clouds
    //             size_t num_available_clouds = keys.size();
    //             size_t start_index = (num_available_clouds > kNumCloudsToViz) ? (num_available_clouds - kNumCloudsToViz) : 0;
                
    //             // 3. Loop from the (N-5)th cloud to the (N)th cloud
    //             size_t cloud_viz_index = 0; // This will be our ID suffix (0 to 4)
    //             for (size_t i = start_index; i < num_available_clouds; ++i) {
    //                 uint64_t key = keys[i];
    //                 const auto& cloud_data = vis_data_ptr->points->at(key);
    //                 std::string cloud_id = "point_cloud_" + std::to_string(cloud_viz_index);
                    
    //                 pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>());
    //                 vg.setInputCloud(cloud_data.points);
    //                 vg.filter(*cloud_filtered);

    //                 // Check if this is the *most recent* cloud in the list
    //                 if (i == num_available_clouds - 1) {
    //                     // --- Display Last Cloud (Colored by Intensity) ---
    //                     pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_handler(cloud_filtered, "intensity");
    //                     if (!viewer->updatePointCloud(cloud_filtered, intensity_handler, cloud_id)) {
    //                         viewer->addPointCloud(cloud_filtered, intensity_handler, cloud_id);
    //                         viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.0, cloud_id);
    //                     }
    //                 } else {
    //                     // --- Display Older Clouds (Colored Gray) ---
    //                     pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> color_handler(cloud_filtered, 128, 128, 128); 
    //                     if (!viewer->updatePointCloud(cloud_filtered, color_handler, cloud_id)) {
    //                         viewer->addPointCloud(cloud_filtered, color_handler, cloud_id);
    //                         viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, cloud_id);
    //                     }
    //                 }
    //                 cloud_viz_index++;
    //             }
    //         }

    //         // --- 3. Update Trajectories (Your Existing Code) ---
    //         ins_trajectory_cloud->clear();
    //         if (vis_data_ptr->insposes) {
    //             for (const auto& key_value : *(vis_data_ptr->insposes)) {
    //                 const auto& pose_matrix = key_value.second.pose;
    //                 pcl::PointXYZRGB ins_point;
    //                 ins_point.x = pose_matrix(0, 3);
    //                 ins_point.y = pose_matrix(1, 3);
    //                 ins_point.z = pose_matrix(2, 3);
    //                 ins_point.r = 30;
    //                 ins_point.g = 144;
    //                 ins_point.b = 255;
    //                 ins_trajectory_cloud->push_back(ins_point);
    //             }
    //         }
    //         if (!viewer->updatePointCloud(ins_trajectory_cloud, "ins_trajectory_cloud")) {
    //             viewer->addPointCloud(ins_trajectory_cloud, "ins_trajectory_cloud");
    //             viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "ins_trajectory_cloud");
    //         }

    //         lo_trajectory_cloud->clear();
    //         if (vis_data_ptr->loposes) {
    //             for (const auto& key_value : *(vis_data_ptr->loposes)) {
    //                 const auto& pose_matrix = key_value.second.pose;
    //                 pcl::PointXYZRGB lo_point;
    //                 lo_point.x = pose_matrix(0, 3);
    //                 lo_point.y = pose_matrix(1, 3);
    //                 lo_point.z = pose_matrix(2, 3);
    //                 lo_point.r = 255;
    //                 lo_point.g = 20;
    //                 lo_point.b = 147;
    //                 lo_trajectory_cloud->push_back(lo_point);
    //             }
    //         }
    //         if (!viewer->updatePointCloud(lo_trajectory_cloud, "lo_trajectory_cloud")) {
    //             viewer->addPointCloud(lo_trajectory_cloud, "lo_trajectory_cloud");
    //             viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "lo_trajectory_cloud");
    //         }

    //         // --- 4. Update Camera Position (Your Existing Code) ---
    //         current_focal_point = current_focal_point + (target_focal_point - current_focal_point) * kSmoothingFactor;
    //         Eigen::Vector3d target_cam_pos = current_focal_point + kCameraOffset;
    //         current_cam_pos = current_cam_pos + (target_cam_pos - current_cam_pos) * kSmoothingFactor;
    //          viewer->setCameraPosition(
    //             current_cam_pos.x(), current_cam_pos.y(), current_cam_pos.z(),
    //             current_focal_point.x(), current_focal_point.y(), current_focal_point.z(),
    //             kUpVector.x(), kUpVector.y(), kUpVector.z()
    //         );

    //         // --- 5. Return data to pool (CRITICAL) ---
    //         std::cout << "Vis data pool size: " << vis_data_pool.GetAvailableCount() << std::endl; 
    //         vis_data_pool.Return(std::move(vis_data_ptr));

    //         // --- 6. Spin viewer ---
    //         viewer->spinOnce(100);
    //     }
    //     std::cout << "Visualization thread exiting\n";
    // });

    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    lidar_socket->stop();
    comp_socket->stop();

    comp_iocontext.stop();
    lidar_iocontext.stop();

    packetCompQueue.stop();
    packetLidarQueue.stop();
    frameLidarQueue.stop();
    frameCompQueue.stop();
    frameDataQueue.stop();
    visDataQueue.stop();
    
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
    std::cout << "Joined comp_processing_thread." << std::endl;

    std::cout << "Joining sync_thread..." << std::endl;
    if (sync_thread.joinable()) sync_thread.join();
    std::cout << "Joined sync_thread." << std::endl;

    std::cout << "Joining lo_thread..." << std::endl;
    if (lo_thread.joinable()) lo_thread.join();
    std::cout << "Joined lo_thread." << std::endl;
    // if (viz_thread.joinable()) viz_thread.join();

    std::cout << "All threads have been joined. Shutdown complete." << std::endl;
}

