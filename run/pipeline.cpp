#include <pipeline.hpp>

static std::atomic<bool> running = true;

void signal_handler(int) {
    running = false;
}

int main() {

    struct sigaction sa = {};
    sa.sa_handler = signal_handler;
    sigaction(SIGINT, &sa, nullptr);

    std::string lidarmeta = "../config/lidar_meta_berlin.json";
    std::string lidarparam = "../config/lidar_config_berlin.json";
    std::string imuparam = "../config/imu_config_berlin.json";
    std::string registerparam = "../config/register_config.json";
    LidarCallback lidarCallback(lidarmeta, lidarparam);                 // initialize
    CompCallback compCallback(imuparam);
    RegisterCallback registerCallback(registerparam);                                // initialize
    RegisterCallback loopcCallback(registerparam);  

    FrameQueue<LidarFrame> lidarQueue;
    FrameQueue<std::deque<CompFrame>> compQueue;
    FrameQueue<FrameData> dataQueue;
    FrameQueue<VisualizationData> vizQueue;

    std::shared_ptr<std::deque<CompFrame>> compWin = std::make_shared<std::deque<CompFrame>>(); // Changed to shared_ptr
    
    auto lidarLastID = std::make_shared<uint16_t>(0);
    auto compLastTs = std::make_shared<double>(0);
    auto compWinSize = std::make_shared<size_t>(20);
    auto keyLidarTs = std::make_shared<double>(0);

    boost::asio::io_context lidar_iocontext;
    boost::asio::io_context comp_iocontext;

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
    //####################################################################################################
    auto lidar_callback = [&lidarCallback, &lidarQueue, lidarLastID](const DataBuffer& packet) {
        if (!running) return;
        auto frame = std::make_unique<LidarFrame>();
        lidarCallback.DecodePacketRng19(packet, *frame);
        
        if (frame->numberpoints > 0 && frame->frame_id != *lidarLastID) {
            *lidarLastID = frame->frame_id;
            // std::cout << "Decoded frame " << frame->frame_id << " with " << frame->numberpoints << " points\n";
            // Move the frame pointer into the queue. No heavy copying.
            lidarQueue.push(std::move(frame));
        }};
    //####################################################################################################
    auto comp_callback = [&compCallback, &compQueue, compLastTs, compWin, compWinSize](const DataBuffer& packet) {
        if (!running) return;
        auto frame = std::make_unique<CompFrame>();
        compCallback.Decode(packet, *frame); // Assuming Decode is a static method or external function
        if (frame->timestamp_20 > 0 && frame->timestamp_20 != *compLastTs) {
            *compLastTs = frame->timestamp_20;
            
            if (compWin->size() >= *compWinSize) {
                compWin->pop_front();
            }
            compWin->push_back(*frame); // Copy CompFrame into compWin
            // std::cout << "Decoded compass frame with timestamp " << frame->timestamp << ".\n";
            // std::cout << "Compass window size: " << compWin->size() << ".\n";
            // Move the entire compWin into the queue when it reaches compWinSize
            if (compWin->size() == *compWinSize) {
                auto compWinCopy = std::make_unique<std::deque<CompFrame>>(*compWin); // Create a copy of compWin
                compQueue.push(std::move(compWinCopy)); // Push the copy into compQueue
            }
        }};
    //####################################################################################################
    auto lidar_errcallback = [](const boost::system::error_code& ec) {
        if (running) {
            std::cerr << "LiDAR IO error: " << ec.message() << " (code: " << ec.value() << ")\n";
        }};
    //####################################################################################################
    auto comp_errcallback = [](const boost::system::error_code& ec) {
        if (running) {
            std::cerr << "Compass IO error: " << ec.message() << " (code: " << ec.value() << ")\n";
        }};
    //####################################################################################################
    auto lidar_socket = std::shared_ptr<UdpSocket>(UdpSocket::create(lidar_iocontext, lidarUdpConfig, lidar_callback, lidar_errcallback));
    //####################################################################################################
    auto comp_socket = std::shared_ptr<UdpSocket>(UdpSocket::create(comp_iocontext, compUdpConfig, comp_callback, comp_errcallback));
    //####################################################################################################
    auto lidar_iothread = std::thread([&lidar_iocontext, &lidarQueue]() {
        try {
            while (running) {
                lidar_iocontext.run_one();
            }
        } catch (const std::exception& e) {
            std::cerr << "LiDAR IO context error: " << e.what() << "\n";
        }
        lidarQueue.stop();
    });
    //####################################################################################################
    auto comp_iothread = std::thread([&comp_iocontext, &compQueue]() {
        try {
            while (running) {
                comp_iocontext.run_one();
            }
        } catch (const std::exception& e) {
            std::cerr << "Compass IO context error: " << e.what() << "\n";
        }
        compQueue.stop();
    });
    //####################################################################################################
    auto sync_thread = std::thread([&lidarQueue, &compQueue, &dataQueue, keyLidarTs]() {
        std::unique_ptr<std::deque<CompFrame>> current_comp_window = nullptr;
        bool is_first_frame = true;

        // Define the interpolation lambda outside the main loop for clarity.
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
                    return a.linearInterpolate(a, b, t); // Assumes linearInterpolate is a const method
                }
            }
            return window->back();
        };

        try {
            while (running) {
                auto lidar_frame = lidarQueue.pop();
                if (!lidar_frame) {
                    if (!running) std::cout << "LiDAR queue stopped, exiting sync thread.\n";
                    break;
                }

                if (lidar_frame->timestamp_points.size() < 2) { //
                    std::cerr << "LiDAR frame " << lidar_frame->frame_id << " has insufficient points, skipping.\n";
                    continue;
                }

                const double max_lidar_time = lidar_frame->timestamp_points.back(); //

                if (is_first_frame) { //
                    *keyLidarTs = max_lidar_time; //
                    is_first_frame = false; //
                    // std::cout << std::setprecision(12) << "Initialized keyLidarTs with first frame: " << *keyLidarTs << std::endl;
                    continue; //
                }

                const double start_interval = *keyLidarTs; //
                const double end_interval = max_lidar_time; //

                if (end_interval <= start_interval) { //
                    std::cerr << "LiDAR frame " << lidar_frame->frame_id << " is out of order or redundant, skipping.\n";
                    continue;
                }

                bool data_gap_detected = false; //
                while (running) {
                    if (!current_comp_window) { //
                        current_comp_window = compQueue.pop(); //
                        if (!current_comp_window) {
                            if (!running) std::cout << "Compass queue stopped, exiting sync thread.\n";
                            break;
                        }
                    }

                    if (current_comp_window->back().timestamp_20 < end_interval) { //
                        // std::cout << std::setprecision(12) << "Compass window not sufficient (ends at " << current_comp_window->back().timestamp
                        //         << ", need to reach " << end_interval << "). Waiting for more data...\n";
                        current_comp_window = nullptr; //
                        continue; //
                    }

                    if (current_comp_window->front().timestamp_20 > start_interval) { //
                        std::cerr << std::setprecision(12) << "CRITICAL: Data gap detected in compass stream. "
                                << "Required interval starts at " << start_interval
                                << " but available data starts at " << current_comp_window->front().timestamp_20 << ".\n";
                        *keyLidarTs = current_comp_window->back().timestamp_20; //
                        current_comp_window = nullptr; //
                        data_gap_detected = true; //
                        break; //
                    }
                    break;
                }

                if (!running || !current_comp_window) {
                    break;
                }

                if (data_gap_detected) { //
                    std::cerr << "Skipping LiDAR frame " << lidar_frame->frame_id << " due to compass data gap.\n";
                    continue; //
                }

                // std::cout << "Aligned LiDAR frame " << lidar_frame->frame_id << std::setprecision(12)
                //         << " (Interval: " << start_interval << " to " << end_interval
                //         << ") with compass window (time: " << current_comp_window->front().timestamp
                //         << " to " << current_comp_window->back().timestamp << ")\n";

                // OPTIMIZATION: Populate dataFrame directly, avoiding intermediate filtCompFrame vector
                auto dataFrame = std::make_unique<FrameData>(); //
                dataFrame->points = lidar_frame->toPCLPointCloud(); //
                dataFrame->timestamp = end_interval; //
                
                // Reserve space once
                dataFrame->ins.reserve(current_comp_window->size() + 2);

                // Add interpolated start point
                CompFrame start_frame = getInterpolated(start_interval, current_comp_window);
                dataFrame->ins.push_back(start_frame); //

                // Add intermediate points
                for (const auto& data : *current_comp_window) {
                    if (data.timestamp_20 > start_interval && data.timestamp_20 < end_interval) { //
                        dataFrame->ins.push_back(start_frame);
                    }
                }

                // Add interpolated end point
                CompFrame end_frame = getInterpolated(end_interval, current_comp_window);
                dataFrame->ins.push_back(end_frame); //

                // std::cout << "Generated compass data with " << dataFrame->imu.size() << " frames for the interval.\n";
                // std::cout << "Imu value for last data frame " << dataFrame->imu.back().acc.transpose() << ".\n";

                dataQueue.push(std::move(dataFrame));
                // std::cout << "Data frame queue size " << dataQueue.size()<< ".\n";

                *keyLidarTs = end_interval; //
            }
        } catch (const std::exception& e) {
            std::cerr << "Sync thread error: " << e.what() << "\n";
        }
        std::cout << "Sync thread exiting\n";
    });
    //####################################################################################################
    // auto ins_viz_thread = std::thread([&registerCallback, &dataQueue]() {
    //     PointsHashMap pointsArchive;
    //     PoseHashMap insPosesArchive;
    //     Eigen::Vector3d rlla = Eigen::Vector3d::Zero();
    //     bool is_first_keyframe = true;
    //     auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("INS Map and Trajectory");
    //     viewer->setBackgroundColor(0.1, 0.1, 0.1);
    //     viewer->addCoordinateSystem(10.0, "world_origin");
    //     viewer->initCameraParameters();
    //     viewer->setCameraPosition(0, 0, -50, 0, 0, 0, 1, 0, 0);

    //     // Static LiDAR-to-body transformation with validation
    //     static Eigen::Matrix4d Tl2b = []() {
    //         double rollAl2b = 0;
    //         double pitchAl2b = 0;
    //         double yawAl2b = 3.141592653589793;
    //         Eigen::AngleAxisd rollAnglel2b(rollAl2b, Eigen::Vector3d::UnitX());
    //         Eigen::AngleAxisd pitchAnglel2b(pitchAl2b, Eigen::Vector3d::UnitY());
    //         Eigen::AngleAxisd yawAnglel2b(yawAl2b, Eigen::Vector3d::UnitZ());
    //         Eigen::Matrix3d Cl2b = (yawAnglel2b * pitchAnglel2b * rollAnglel2b).toRotationMatrix();
    //         if (!Cl2b.allFinite() || std::abs(Cl2b.determinant() - 1.0) > 1e-6) {
    //             throw std::runtime_error("Invalid LiDAR-to-body rotation matrix");
    //         }
    //         Eigen::Vector3d tl2b{0.135, 0.0, -0.1243};
    //         Eigen::Matrix4d Tl2b_init = Eigen::Matrix4d::Identity();
    //         Tl2b_init.block<3,3>(0,0) = Cl2b;
    //         Tl2b_init.block<3,1>(0,3) = tl2b;
    //         return Tl2b_init;
    //     }();

    //     try {
    //         while (running && !viewer->wasStopped()) {
    //             auto data_frame = dataQueue.pop();
    //             if (!data_frame) {
    //                 if (!running) std::cout << "Data queue stopped, exiting INS viz thread.\n";
    //                 break;
    //             }
    //             if (data_frame->position.empty()) {
    //                 std::cerr << "Frame ID: " << data_frame->points.frame_id << " has empty position data, skipping.\n";
    //                 continue;
    //             }
    //             // --- DATA EXTRACTION & POSE CALCULATION ---
    //             pcl::PointCloud<pcl::PointXYZI>::Ptr pointsBody(new pcl::PointCloud<pcl::PointXYZI>());
    //             *pointsBody = std::move(data_frame->points.pointsBody);
    //             const Eigen::Vector3d& lla = data_frame->position.back().pose;

    //             // Use quaternion directly for rotation matrix (avoid redundant Euler computation)
    //             const auto& quat = data_frame->position.back().orientation;
    //             Eigen::Matrix3d Cb2m = quat.cast<double>().toRotationMatrix();

    //             // For debugging: Optionally compute from Euler and compare
    //             const auto& euler_angles_rad = data_frame->position.back().euler.cast<double>();
    //             double roll = euler_angles_rad.x();
    //             double pitch = euler_angles_rad.y();
    //             double yaw = euler_angles_rad.z();
    //             constexpr double RAD_TO_DEG = 180.0 / M_PI;
    //             double roll_deg = euler_angles_rad.x() * RAD_TO_DEG;
    //             double pitch_deg = euler_angles_rad.y() * RAD_TO_DEG;
    //             double yaw_deg = euler_angles_rad.z() * RAD_TO_DEG;
    //             Eigen::Vector3d Eulerdeg{roll_deg, pitch_deg, yaw_deg};
    //             Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    //             Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    //             Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
    //             Eigen::Matrix3d Cb2m_from_euler = (yawAngle * pitchAngle * rollAngle).toRotationMatrix();

    //             if (!Cb2m.allFinite() || std::abs(Cb2m.determinant() - 1.0) > 1e-6) {
    //                 std::cerr << "Frame ID: " << data_frame->points.frame_id << " has invalid orientation matrix, skipping.\n";
    //                 continue;
    //             }
    //             uint64_t id = data_frame->points.frame_id;
    //             Eigen::Vector3d tb2m = Eigen::Vector3d::Zero();
    //             if (is_first_keyframe) {
    //                 rlla = lla;
    //                 // tb2m = registerCallback.lla2ned(lla.x(), lla.y(), lla.z(), rlla.x(), rlla.y(), rlla.z());
    //                 is_first_keyframe = false;
    //             } else {
    //                 tb2m = registerCallback.lla2ned(lla.x(), lla.y(), lla.z(), rlla.x(), rlla.y(), rlla.z());
    //             }
    //             if (!tb2m.allFinite()) {
    //                 std::cerr << "Frame ID: " << id << " has invalid NED coordinates, skipping.\n";
    //                 continue;
    //             }
    //             Eigen::Matrix4d Tb2m = Eigen::Matrix4d::Identity();
    //             Tb2m.block<3,3>(0,0) = Cb2m;
    //             Tb2m.block<3,1>(0,3) = tb2m;
    //             Eigen::Matrix4d Tl2m = Tb2m * Tl2b;
    //             pcl::PointCloud<pcl::PointXYZI>::Ptr pointsMap(new pcl::PointCloud<pcl::PointXYZI>());
    //             pcl::transformPointCloud(*pointsBody, *pointsMap, Tl2m.cast<float>());

    //             std::cout << std::fixed << ".................................................." << std::endl;
    //             std::cout << std::fixed << "Viz Thread.............................." << std::endl;
    //             std::cout << std::fixed << "Frame ID................................" << id << std::endl;
    //             std::cout << std::fixed << "Number points..........................." << pointsBody->size() << std::endl;
    //             std::cout << std::fixed << std::setprecision(12) << "LLA.....................................\n" << lla.transpose() << std::endl;
    //             std::cout << std::fixed << std::setprecision(12) << "From Euler Deg..........................\n" << Eulerdeg.transpose() << std::endl;
    //             std::cout << std::fixed << "Cb2m Euler (ZYX)........................\n" << Cb2m_from_euler << std::endl;
    //             std::cout << std::fixed << "Cb2m....................................\n" << Cb2m << std::endl;
    //             std::cout << std::fixed << "tb2m....................................\n" << tb2m.transpose() << std::endl;
    //             std::cout << std::fixed << "Tl2m....................................\n" << Tl2m << std::endl;
    //             std::cout << std::fixed << ".................................................." << std::endl;

    //             // Eigen::Matrix4d Tm2b = Tb2m.inverse();

    //             // --- DATA ARCHIVING ---
    //             // Remove clear() to accumulate full map
    //             // pointsArchive.clear();
    //             pointsArchive[id] = {pointsMap, data_frame->timestamp};
    //             insPosesArchive[id] = {Tb2m, data_frame->timestamp};

    //             // --- VISUALIZATION ---
    //             viewer->removeAllPointClouds();

    //             // Aggregate and display the full accumulated map (downsampled)
    //             pcl::PointCloud<pcl::PointXYZI>::Ptr aggregatedMap(new pcl::PointCloud<pcl::PointXYZI>());
    //             for (const auto& kv : pointsArchive) {
    //                 *aggregatedMap += *(kv.second.points);
    //             }
    //             pcl::PointCloud<pcl::PointXYZI>::Ptr aggregatedMapDS(new pcl::PointCloud<pcl::PointXYZI>());
    //             if (!aggregatedMap->empty()) {
    //                 pcl::VoxelGrid<pcl::PointXYZI> vg;
    //                 vg.setLeafSize(1.0f, 1.0f, 1.0f);
    //                 vg.setInputCloud(aggregatedMap);
    //                 vg.filter(*aggregatedMapDS);
    //                 // for (auto& point : aggregatedMapDS->points) {
    //                 //     point.x = point.x;    // 
    //                 //     point.y = point.y; // 
    //                 //     point.z = point.z;   // 
    //                 // }
    //             }
    //             if (!aggregatedMapDS->empty()) {
    //                 pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> map_color_handler(aggregatedMapDS, "intensity");
    //                 viewer->addPointCloud(aggregatedMapDS, map_color_handler, "map_cloud");
    //                 viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "map_cloud");
    //             }

    //             // Fix: Remove old coordinate system before adding new one (for latest pose)
    //             // viewer->removeCoordinateSystem("vehicle_pose");
    //             // Eigen::Affine3f vehicle_pose = Eigen::Affine3f::Identity();
    //             // vehicle_pose.matrix() = Tb2m.cast<float>();
    //             // viewer->addCoordinateSystem(3.0, vehicle_pose, "vehicle_pose");

    //             // Display the full accumulated trajectory
    //             // pcl::PointCloud<pcl::PointXYZRGB>::Ptr trajectory_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    //             // for (const auto& kv : insPosesArchive) {
    //             //     // Get a const reference to the original pose
    //             //     const auto& original_pose = kv.second.pose;

    //             //     // Declare a NEW matrix to store the inverted result
    //             //     // Eigen::Matrix4d inverted_pose = original_pose.inverse().eval();

    //             //     // Now, use the new 'inverted_pose' variable
    //             //     pcl::PointXYZRGB point;
    //             //     point.x = original_pose(0, 3);
    //             //     point.y = original_pose(1, 3);
    //             //     point.z = original_pose(2, 3);
    //             //     point.r = 255; point.g = 10; point.b = 10;
    //             //     trajectory_cloud->push_back(point);
    //             // }
    //             // if (!trajectory_cloud->empty()) {
    //             //     viewer->addPointCloud(trajectory_cloud, "trajectory_cloud");
    //             //     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "trajectory_cloud");
    //             // }
    //             viewer->spinOnce(100);
    //         }
    //     } catch (const std::exception& e) {
    //         std::cerr << "INS viz thread error: " << e.what() << "\n";
    //     }
    //     std::cout << "INS viz thread exiting\n";
    // });

    //####################################################################################################
    auto gtsam_thread = std::thread([&registerCallback, &dataQueue, &vizQueue]() {
        // --- MAPS AND NDT ARE NOW DECLARED AND OWNED BY THIS THREAD ---
        const double VOXEL_SIZE = 5.0; 
        const double LOOP_CLOSURE_TIME_THRESHOLD = 180.0;
        const int NEIGHBOR_SEARCH_SIZE = 1;

        PointsHashMap pointsArchive;
        VoxelHashMap spatialArchive;

        Eigen::Vector3d rlla  = Eigen::Vector3d::Zero(); 
        Eigen::Matrix4d lidarFactorSourceTb2m = Eigen::Matrix4d::Identity();
        // Eigen::Vector<double, 6> lidarCovScalingVector{10, 1, 1, 1e3, 1e3, 1e3}; // Default/high-trust scaling
        Eigen::Vector<double, 6> low_trust_lidar_scaling_vector{1e3, 1e3, 1e3, 1e2, 1e2, 1e2}; // Scaling for poor-quality matches
        
        // Trust Gain parameters defined here ---
        Eigen::Vector<double, 6> insCovScalingVector{1e3, 1e3, 1e3, 1e3, 1e3, 1e3}; // High uncertainty for denied state
        bool was_gps_denied = false; // Assume we start in a denied state
        double current_trust_factor = 0.0;
        const double recovery_rate = 0.01; // Trust regained over 1/0.02 = 5 keyframes
        const Eigen::Vector<double, 6> full_trust_scaling_vector = Eigen::Vector<double, 6>::Ones(); // Target is 1.0 scaling

        pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>::Ptr ndt_omp = nullptr;
        if (registerCallback.registration_method_ == "NDT") {
            ndt_omp.reset(new pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>());
            ndt_omp->setNumThreads(registerCallback.num_threads_);
            ndt_omp->setResolution(registerCallback.ndt_resolution_);
            ndt_omp->setTransformationEpsilon(registerCallback.ndt_transform_epsilon_);

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
            registerCallback.registration = ndt_omp;
        } 

        bool is_first_keyframe = true;
        uint64_t last_id = 0;
        gtsam::ISAM2Params isam2_params;
        isam2_params.relinearizeThreshold = 0.1;
        isam2_params.relinearizeSkip = 1;
        gtsam::ISAM2 isam2(isam2_params);
        gtsam::Values Val;

        try {
            while (running) {
                auto data_frame = dataQueue.pop();
                if (!data_frame) {
                    if (!running) std::cout << "Factor queue stopped, exiting Gtsam thread.\n";
                    break;
                }

                Eigen::Matrix<double, 6, 6> lidarCov = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
                Eigen::Matrix<double, 6, 6> loopCov = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
                Eigen::Vector<double, 6> lidarStdDev = Eigen::Vector<double, 6>::Zero();
                Eigen::Vector<double, 6> insStdDev = Eigen::Vector<double, 6>::Zero();
                Eigen::Vector<double, 6> insScaledStdDev = Eigen::Vector<double, 6>::Zero();

                pcl::PointCloud<pcl::PointXYZI>::Ptr pointsBody(new pcl::PointCloud<pcl::PointXYZI>());
                *pointsBody = std::move(data_frame->points.pointsBody);
                auto key_data_frame = data_frame->ins.back();
                const Eigen::Vector3d lla{key_data_frame.latitude_20, key_data_frame.longitude_20, key_data_frame.altitude_20};
                Eigen::Quaternionf quat{key_data_frame.qw_20, key_data_frame.qx_20, key_data_frame.qy_20, key_data_frame.qz_20};
                const Eigen::Matrix3d Cb2m = quat.toRotationMatrix().cast<double>();
                Eigen::Matrix4d Tb2m = Eigen::Matrix4d::Identity();
                insStdDev << key_data_frame.sigmaLatitude_20, key_data_frame.sigmaLongitude_20, key_data_frame.sigmaAltitude_20, key_data_frame.sigmaRoll_26, key_data_frame.sigmaPitch_26, key_data_frame.sigmaYaw_26;
                insStdDev = insStdDev * 0.1;
                uint64_t id = data_frame->points.frame_id;
                double timestamp = data_frame->timestamp;
                int ndt_iter = 0;
                std::chrono::milliseconds align_duration;

                gtsam::NonlinearFactorGraph newFactors;
                gtsam::Values newEstimates;

                if (is_first_keyframe) {
                    rlla = lla;
                    Eigen::Vector3d tb2m = registerCallback.lla2ned(lla.x(),lla.y(),lla.z(),rlla.x(),rlla.y(),rlla.z());
                    Tb2m = Eigen::Matrix4d::Identity();
                    Tb2m.block<3,3>(0,0) = Cb2m.cast<double>();
                    Tb2m.block<3,1>(0,3) = tb2m;

                    gtsam::Pose3 insFactor(Tb2m);
                    gtsam::SharedNoiseModel insNoiseModel = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << insStdDev(3), insStdDev(4), insStdDev(5), insStdDev(0), insStdDev(1), insStdDev(2)).finished());
                    newEstimates.insert(Symbol('x', id), insFactor);
                    newFactors.add(gtsam::PriorFactor<gtsam::Pose3>(Symbol('x', id), std::move(insFactor), std::move(insNoiseModel)));

                } else {
                    Eigen::Vector3d tb2m = registerCallback.lla2ned(lla.x(),lla.y(),lla.z(),rlla.x(),rlla.y(),rlla.z());
                    Tb2m = Eigen::Matrix4d::Identity();
                    Tb2m.block<3,3>(0,0) = Cb2m.cast<double>();
                    Tb2m.block<3,1>(0,3) = tb2m;

                    gtsam::Pose3 initialFactor(lidarFactorSourceTb2m);
                    newEstimates.insert(Symbol('x', id), initialFactor);
                    pcl::PointCloud<pcl::PointXYZI>::Ptr lidarFactorPointsSource(new pcl::PointCloud<pcl::PointXYZI>());
                    pcl::PointCloud<pcl::PointXYZI>::Ptr lidarFactorPointsTarget(new pcl::PointCloud<pcl::PointXYZI>());
                    const auto& lidarFactorPointsArchive = pointsArchive.at(last_id);
                    gtsam::Pose3 lidarFactorTargetTb2m = Val.at<gtsam::Pose3>(Symbol('x', last_id));
                    pcl::transformPointCloud(*lidarFactorPointsArchive.points, *lidarFactorPointsTarget, lidarFactorTargetTb2m.matrix());
                    registerCallback.registration->setInputTarget(lidarFactorPointsTarget);
                    registerCallback.registration->setInputSource(pointsBody);
                    auto align_start = std::chrono::high_resolution_clock::now();
                    registerCallback.registration->align(*lidarFactorPointsSource, lidarFactorSourceTb2m.cast<float>());
                    auto align_end = std::chrono::high_resolution_clock::now();
                    align_duration = std::chrono::duration_cast<std::chrono::milliseconds>(align_end - align_start);
                    if (registerCallback.registration->hasConverged()) {
                        std::cout << "Logging: NDT converged. Using scaled covariance.\n";
                        lidarFactorSourceTb2m = registerCallback.registration->getFinalTransformation().cast<double>();
                        Eigen::Matrix4d LidarTbs2bt = lidarFactorTargetTb2m.matrix().inverse()*lidarFactorSourceTb2m;
                        auto ndt_result = ndt_omp->getResult();
                        ndt_iter = ndt_result.iteration_num;
                        const auto& hessian = ndt_result.hessian;
                        Eigen::Matrix<double, 6, 6> regularized_hessian = hessian + (Eigen::Matrix<double, 6, 6>::Identity() * 1e-6);
                        lidarCov = -regularized_hessian.inverse();
                        lidarStdDev = lidarCov.diagonal().cwiseSqrt();
                        
                        gtsam::Pose3 lidarFactor = gtsam::Pose3(std::move(LidarTbs2bt));
                        gtsam::SharedNoiseModel lidarNoiseModel = gtsam::noiseModel::Gaussian::Covariance(registerCallback.reorderCovarianceForGTSAM(std::move(lidarCov)));
                        newFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(Symbol('x', last_id), Symbol('x', id), std::move(lidarFactor), std::move(lidarNoiseModel)));

                    } else {
                        std::cout << "Warning: NDT not converged. Using low-trust covariance.\n";
                        Eigen::Matrix4d LidarTbs2bt = lidarFactorTargetTb2m.matrix().inverse()*lidarFactorSourceTb2m;
                        gtsam::Pose3 lidarFactor = gtsam::Pose3(LidarTbs2bt);
                        Eigen::Matrix<double, 6, 6> covariance;
                        lidarCov = low_trust_lidar_scaling_vector.asDiagonal();
                        lidarStdDev = lidarCov.diagonal().cwiseSqrt();
                        gtsam::SharedNoiseModel lidarNoiseModel = gtsam::noiseModel::Gaussian::Covariance(registerCallback.reorderCovarianceForGTSAM(std::move(lidarCov)));
                        newFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(Symbol('x', last_id), Symbol('x', id), std::move(lidarFactor), std::move(lidarNoiseModel)));
                    }

                    bool is_gps_available_now = (insStdDev.norm() < 1.0);
                    if (is_gps_available_now && was_gps_denied) {
                        std::cout << "Warning: GPS return from denied position.start trust gain recovery.\n";
                        current_trust_factor = 0.0; // Reset to begin recovery from zero trust
                    }
                    was_gps_denied = !is_gps_available_now;
                    Eigen::Vector<double, 6> current_ins_scaling_vector;
                    if (is_gps_available_now) {
                        // If available, increase trust factor and interpolate the scaling vector.
                        current_trust_factor = std::min(1.0, current_trust_factor + recovery_rate);
                        current_ins_scaling_vector = insCovScalingVector + current_trust_factor * (full_trust_scaling_vector - insCovScalingVector);
                        std::cout << "Logging: GPS Available. Current ins scalling factor.\n" << current_ins_scaling_vector.transpose() << std::endl;;
                    } else {
                        // If denied, reset trust and use the high uncertainty scaling.
                        std::cout << "Warning: GPS Denied. Using low-trust covariance.\n";
                        current_trust_factor = 0.0;
                        current_ins_scaling_vector = insCovScalingVector;
                    }
                    gtsam::Vector6 scaled_sigmas;
                    scaled_sigmas << insStdDev(3) * current_ins_scaling_vector(3), // roll
                                     insStdDev(4) * current_ins_scaling_vector(4), // pitch
                                     insStdDev(5) * current_ins_scaling_vector(5), // yaw
                                     insStdDev(0) * current_ins_scaling_vector(0), // x (from lat)
                                     insStdDev(1) * current_ins_scaling_vector(1), // y (from lon)
                                     insStdDev(2) * current_ins_scaling_vector(2);  // z (from alt)
                    insScaledStdDev << scaled_sigmas(3),scaled_sigmas(4),scaled_sigmas(5),scaled_sigmas(0),scaled_sigmas(1),scaled_sigmas(2);
                    gtsam::Pose3 insFactor(Tb2m);
                    gtsam::SharedNoiseModel insNoiseModel = gtsam::noiseModel::Diagonal::Sigmas(scaled_sigmas);
                    newFactors.add(gtsam::PriorFactor<gtsam::Pose3>(Symbol('x', id), std::move(insFactor), std::move(insNoiseModel)));
                }

                // ###########LOOP CLOSURE
                // bool loopCandidateFound = false;
                // KeyframeInfo loopTargetCandidate = {0, 0.0};
                // gtsam::Pose3 loopFactorSourceTb2m, loopFactorTargetTb2m;
                // if (!is_first_keyframe) {
                //     loopFactorSourceTb2m = Val.at<gtsam::Pose3>(Symbol('x', id - 1));
                //     Voxel loopFactorSourceVoxel = Voxel::getKey(loopFactorSourceTb2m.translation().cast<float>(), VOXEL_SIZE);
                //     double min_distance_sq = std::numeric_limits<double>::max();
                //     for (int dx = -NEIGHBOR_SEARCH_SIZE; dx <= NEIGHBOR_SEARCH_SIZE; ++dx) {
                //         for (int dy = -NEIGHBOR_SEARCH_SIZE; dy <= NEIGHBOR_SEARCH_SIZE; ++dy) {
                //             for (int dz = -NEIGHBOR_SEARCH_SIZE; dz <= NEIGHBOR_SEARCH_SIZE; ++dz) {
                //                 Voxel loopFactorQueryVoxel{loopFactorSourceVoxel.x + dx, loopFactorSourceVoxel.y + dy, loopFactorSourceVoxel.z + dz};
                //                 auto it = spatialArchive.find(loopFactorQueryVoxel);
                //                 if (it != spatialArchive.end()) {
                //                     for (const auto& loopFactorCandidate : it->second) {
                //                         if (std::abs(timestamp - loopFactorCandidate.timestamp) < LOOP_CLOSURE_TIME_THRESHOLD) {
                //                             continue; 
                //                         }
                //                         if (Val.exists(Symbol('x', loopFactorCandidate.id))) {
                //                             loopFactorTargetTb2m = Val.at<gtsam::Pose3>(Symbol('x', loopFactorCandidate.id));
                //                             double dist_sq = (loopFactorSourceTb2m.translation() - loopFactorTargetTb2m.translation()).squaredNorm();
                //                             if (dist_sq < min_distance_sq) {
                //                                 min_distance_sq = dist_sq;
                //                                 loopTargetCandidate = loopFactorCandidate;
                //                                 loopCandidateFound = true;
                //                             }
                //                         }
                //                     }
                //                 }
                //             }
                //         }
                //     }
                // }

                // if (loopCandidateFound) {
                //     std::cout << "Found loop closure.";
                //     const auto& loopFactorSourcePointsArchive = pointsArchive.at(id - 1);
                //     const auto& loopFactorTargetPointsArchive = pointsArchive.at(loopTargetCandidate.id);
                //     pcl::PointCloud<pcl::PointXYZI>::Ptr loopFactorPointsTarget(new pcl::PointCloud<pcl::PointXYZI>());
                //     pcl::PointCloud<pcl::PointXYZI>::Ptr loopFactorPointsSource(new pcl::PointCloud<pcl::PointXYZI>());
                //     pcl::transformPointCloud(*loopFactorTargetPointsArchive.points,*loopFactorPointsTarget,loopFactorTargetTb2m.matrix());

                //     registerCallback.registration->setInputTarget(loopFactorPointsTarget);
                //     registerCallback.registration->setInputSource(loopFactorSourcePointsArchive.points);
                //     registerCallback.registration->align(*loopFactorPointsSource, loopFactorSourceTb2m.matrix().cast<float>());
                //     if (registerCallback.registration->hasConverged()) {
                //         Eigen::Matrix4d loopFactorUpdatedSourceTb2m = registerCallback.registration->getFinalTransformation().cast<double>();
                //         Eigen::Matrix4d loopTbs2bt = loopFactorTargetTb2m.matrix().inverse()*loopFactorUpdatedSourceTb2m;
                //         if (ndt_omp) {
                //             auto ndt_result = ndt_omp->getResult();
                //             const auto& hessian = ndt_result.hessian;
                //             Eigen::Matrix<double, 6, 6> regularized_hessian = hessian + (Eigen::Matrix<double, 6, 6>::Identity() * 1e-6);
                //             if (regularized_hessian.determinant() > 1e-6) {
                //                 loopCov = -regularized_hessian.inverse();
                //             }
                //         }
                //         gtsam::Pose3 loopFactor = gtsam::Pose3(std::move(loopTbs2bt));
                //         gtsam::SharedNoiseModel loopNoiseModel = gtsam::noiseModel::Gaussian::Covariance(registerCallback.reorderCovarianceForGTSAM(std::move(loopCov)));
                //         newFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(Symbol('x', loopTargetCandidate.id), Symbol('x', id - 1), std::move(loopFactor), std::move(loopNoiseModel)));
                //     }
                // }
                // ########################
                isam2.update(newFactors, newEstimates);
                
                // Periodically print a more detailed summary
                Val = isam2.calculateEstimate();

                gtsam::Pose3 currTb2m = Val.at<gtsam::Pose3>(Symbol('x', id));

                if (!is_first_keyframe) {
                    gtsam::Pose3 prevTb2m = Val.at<gtsam::Pose3>(Symbol('x', last_id));
                    Eigen::Matrix4d loopTbc2bp = prevTb2m.matrix().inverse() * currTb2m.matrix();
                    lidarFactorSourceTb2m = currTb2m.matrix() * loopTbc2bp;
                } else {
                    lidarFactorSourceTb2m = currTb2m.matrix();
                    is_first_keyframe = false;
                }
                // #################add single info into spatial map if loop closure not found
                // if (loopCandidateFound){
                    // spatialArchive.clear();
                    // for (const auto& key_value : Val) {
                    //     uint64_t frame_id = gtsam::Symbol(key_value.key).index();
                    //     gtsam::Pose3 pose = key_value.value.cast<gtsam::Pose3>();
                    //     Voxel key = Voxel::getKey(pose.translation().cast<float>(), VOXEL_SIZE);
                    //     spatialArchive[key].push_back({frame_id, pointsArchive.at(frame_id).timestamp});
                    //  }
                // } else {
                    Voxel key = Voxel::getKey(currTb2m.translation().cast<float>(), VOXEL_SIZE);
                    spatialArchive[key].push_back({id, timestamp});
                // }
                pointsArchive[id] = {pointsBody, timestamp};
                last_id = id; 

                if (!Val.empty()) {
                    auto vizData = std::make_unique<VisualizationData>();
                    vizData->poses = std::make_shared<gtsam::Values>(Val);
                    // Pass a deep copy of the points archive for thread safety
                    vizData->points = std::make_shared<PointsHashMap>(pointsArchive);
                    vizQueue.push(std::move(vizData));
                }
                // ################# rebuild spatial map if loop closure found
                std::cout << std::fixed << ".................................................." << std::endl;
                std::cout << std::fixed << "Gtsam Thread............................" << std::endl;
                std::cout << std::fixed << "Frame ID................................" << id << std::endl;
                std::cout << std::fixed << "Number points..........................." << pointsBody->size() << std::endl;
                std::cout << std::fixed << "Alignment Time.........................." << align_duration.count() << " ms" << std::endl;
                std::cout << std::fixed << "Number Iteration........................" << ndt_iter << std::endl;
                std::cout << std::fixed << "Ins Std Dev (m, rad)....................\n" << insScaledStdDev.transpose() << std::endl;
                std::cout << std::fixed << "Lidar Std Dev (m, rad)..................\n" << lidarStdDev.transpose() << std::endl;
                std::cout << std::fixed << "New factors added this step............." << newFactors.size() << std::endl;
                std::cout << std::fixed << "Total factors in graph.................." << isam2.size() << std::endl;
                std::cout << std::fixed << "Tb2m Lidar..............................\n" << lidarFactorSourceTb2m << std::endl;
                std::cout << std::fixed << "Tb2m INS................................\n" << Tb2m << std::endl;
                std::cout << std::fixed << "Optimized Tb2m..........................\n" << currTb2m.matrix() << std::endl;
                std::cout << std::fixed << ".................................................." << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Gtsam thread error: " << e.what() << "\n";
        }
        std::cout << "Gtsam thread exiting\n";
    });
    //####################################################################################################
    auto viz_thread = std::thread([&vizQueue]() {
        auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("GTSAM Optimized Map");
        viewer->setBackgroundColor(0.1, 0.1, 0.1);
        viewer->addCoordinateSystem(10.0, "world_origin");
        viewer->initCameraParameters();

        // --- NEW: Camera Following & Smoothing Logic ---
        // This factor controls how quickly the camera catches up to the target.
        // Lower values (e.g., 0.05) are smoother but have more lag.
        // Higher values (e.g., 0.2) are more responsive but can be jumpy.
        const double kSmoothingFactor = 0.1;

        // This defines the camera's position relative to the focal point (view from above).
        const Eigen::Vector3d kCameraOffset(0.0, 0.0, -250.0);

        // The "up" vector for the camera. Your original code used (1,0,0), which is non-standard but preserved here.
        // A more common "up" vector would be (0, -1, 0) for Z-forward or (0, 0, 1) for Y-forward systems.
        const Eigen::Vector3d kUpVector(1.0, 0.0, 0.0);

        // State variables to hold the camera's current and target focus points.
        // Initialize them to the starting view.
        Eigen::Vector3d target_focal_point(0.0, 0.0, 0.0);
        Eigen::Vector3d current_focal_point = target_focal_point;
        Eigen::Vector3d current_cam_pos = target_focal_point + kCameraOffset;

        // --- MODIFIED: The initial setCameraPosition is now managed by the loop ---
        // viewer->setCameraPosition(0, 0, -50, 0, 0, 0, 1, 0, 0); // This is now handled dynamically

        const size_t kSlidingWindowSize = 100;
        std::deque<uint64_t> displayed_frame_ids;
        uint64_t last_processed_id = 0;

        pcl::VoxelGrid<pcl::PointXYZI> vg;
        vg.setLeafSize(0.5f, 0.5f, 0.5f);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr trajectory_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        while (running && !viewer->wasStopped()) {
            auto vizData = vizQueue.pop();
            if (!vizData) {
                if (!running) std::cout << "Visualization queue stopped, exiting viz thread.\n";
                break;
            }

            gtsam::Pose3 latest_pose;
            uint64_t max_id = 0;
            if (!vizData->poses->empty()) { // Ensure poses are not empty before finding max
                for (const auto& key_value : *(vizData->poses)) {
                    uint64_t frame_id = gtsam::Symbol(key_value.key).index();
                    if (frame_id > max_id) {
                        max_id = frame_id;
                    }
                }
            }
            
            if (max_id > 0 && max_id != last_processed_id) {
                last_processed_id = max_id;
                latest_pose = vizData->poses->at<gtsam::Pose3>(Symbol('x', max_id));
                
                // --- NEW: Update the target focal point when a new pose is available ---
                target_focal_point = latest_pose.translation();

                auto it = vizData->points->find(max_id);
                if (it != vizData->points->end()) {
                    // ... (your existing point cloud processing logic is unchanged)
                    if (displayed_frame_ids.size() >= kSlidingWindowSize) {
                        uint64_t id_to_remove = displayed_frame_ids.front();
                        std::string cloud_id_to_remove = "map_cloud_" + std::to_string(id_to_remove);
                        if (viewer->contains(cloud_id_to_remove)) {
                            viewer->removePointCloud(cloud_id_to_remove);
                        }
                        displayed_frame_ids.pop_front();
                    }

                    pcl::PointCloud<pcl::PointXYZI>::Ptr raw_cloud = it->second.points;
                    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>());
                    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZI>());
                    pcl::PointCloud<pcl::PointXYZI>::Ptr spatial_cloud(new pcl::PointCloud<pcl::PointXYZI>());
                    
                    pcl::transformPointCloud(*raw_cloud, *transformed_cloud, latest_pose.matrix().cast<float>());
                    vg.setInputCloud(transformed_cloud);
                    vg.filter(*downsampled_cloud);
                    
                    pcl::PassThrough<pcl::PointXYZI> pass_spatial;
                    pass_spatial.setFilterFieldName("z");
                    pass_spatial.setFilterLimits(-300.0, 0.0);
                    pass_spatial.setInputCloud(downsampled_cloud);
                    pass_spatial.filter(*spatial_cloud);

                    std::string cloud_id_to_add = "map_cloud_" + std::to_string(max_id);
                    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color_handler(spatial_cloud, "intensity");
                    viewer->addPointCloud(spatial_cloud, color_handler, cloud_id_to_add);
                    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloud_id_to_add);
                    
                    displayed_frame_ids.push_back(max_id);
                }
            }

            // ... (your existing trajectory update logic is unchanged)
            trajectory_cloud->clear();
            for (const auto& key_value : *(vizData->poses)) {
                gtsam::Pose3 pose = key_value.value.cast<gtsam::Pose3>();
                pcl::PointXYZRGB trajectory_point;
                trajectory_point.x = pose.translation().x();
                trajectory_point.y = pose.translation().y();
                trajectory_point.z = pose.translation().z();
                trajectory_point.r = 255;
                trajectory_point.g = 10;
                trajectory_point.b = 10;
                trajectory_cloud->push_back(trajectory_point);
            }

            if (!viewer->updatePointCloud(trajectory_cloud, "trajectory_cloud")) {
                viewer->addPointCloud(trajectory_cloud, "trajectory_cloud");
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "trajectory_cloud");
            }

            // --- NEW: Smoothly update camera on every spin ---
            // Interpolate the focal point
            current_focal_point = current_focal_point + (target_focal_point - current_focal_point) * kSmoothingFactor;
            // Calculate the desired camera position based on the smoothed focal point
            Eigen::Vector3d target_cam_pos = current_focal_point + kCameraOffset;
            // Interpolate the camera's actual position
            current_cam_pos = current_cam_pos + (target_cam_pos - current_cam_pos) * kSmoothingFactor;

            // Set the new camera position and view direction in the visualizer
            viewer->setCameraPosition(
                current_cam_pos.x(), current_cam_pos.y(), current_cam_pos.z(),
                current_focal_point.x(), current_focal_point.y(), current_focal_point.z(),
                kUpVector.x(), kUpVector.y(), kUpVector.z()
            );
            
            viewer->spinOnce(100);
        }
        std::cout << "Visualization thread exiting\n";
    });

    //####################################################################################################
    // Cleanup
    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    lidar_socket->stop();
    comp_socket->stop();
    lidarQueue.stop();
    compQueue.stop();
    dataQueue.stop();
    vizQueue.stop();

    if (lidar_iothread.joinable()) lidar_iothread.join();
    if (comp_iothread.joinable()) comp_iothread.join();
    if (sync_thread.joinable()) sync_thread.join();
    if (gtsam_thread.joinable()) gtsam_thread.join();
    if (viz_thread.joinable()) viz_thread.join();
    // if (ins_viz_thread.joinable()) ins_viz_thread.join();
    
    std::cout << "All threads have been joined. Shutdown complete." << std::endl;
}