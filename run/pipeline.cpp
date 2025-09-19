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

    FrameQueue<LidarFrame> lidarQueue;
    FrameQueue<std::deque<CompFrame>> compQueue;
    FrameQueue<FrameData> dataQueue;
    FrameQueue<GtsamFactorData> factorQueue;
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
        if (frame->timestamp > 0 && frame->timestamp != *compLastTs) {
            *compLastTs = frame->timestamp;
            
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
            if (target_time <= window->front().timestamp) return window->front();
            if (target_time >= window->back().timestamp) return window->back();
            for (size_t i = 0; i < window->size() - 1; ++i) {
                const CompFrame& a = (*window)[i];
                const CompFrame& b = (*window)[i + 1];
                if (a.timestamp <= target_time && target_time <= b.timestamp) {
                    double t = (b.timestamp - a.timestamp > 1e-9) ? (target_time - a.timestamp) / (b.timestamp - a.timestamp) : 0.0;
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

                    if (current_comp_window->back().timestamp < end_interval) { //
                        // std::cout << std::setprecision(12) << "Compass window not sufficient (ends at " << current_comp_window->back().timestamp
                        //         << ", need to reach " << end_interval << "). Waiting for more data...\n";
                        current_comp_window = nullptr; //
                        continue; //
                    }

                    if (current_comp_window->front().timestamp > start_interval) { //
                        std::cerr << std::setprecision(12) << "CRITICAL: Data gap detected in compass stream. "
                                << "Required interval starts at " << start_interval
                                << " but available data starts at " << current_comp_window->front().timestamp << ".\n";
                        *keyLidarTs = current_comp_window->back().timestamp; //
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
                dataFrame->imu.reserve(current_comp_window->size() + 2);
                dataFrame->position.reserve(current_comp_window->size() + 2);

                // Add interpolated start point
                CompFrame start_frame = getInterpolated(start_interval, current_comp_window);
                dataFrame->imu.push_back(start_frame.toImuData()); //
                dataFrame->position.push_back(start_frame.toPositionData()); //

                // Add intermediate points
                for (const auto& data : *current_comp_window) {
                    if (data.timestamp > start_interval && data.timestamp < end_interval) { //
                        dataFrame->imu.push_back(data.toImuData()); //
                        dataFrame->position.push_back(data.toPositionData()); //
                    }
                }

                // Add interpolated end point
                CompFrame end_frame = getInterpolated(end_interval, current_comp_window);
                dataFrame->imu.push_back(end_frame.toImuData()); //
                dataFrame->position.push_back(end_frame.toPositionData()); //

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
    auto factor_thread = std::thread([&registerCallback, &dataQueue, &factorQueue]() {
        bool is_first_keyframe = true;

        Eigen::Matrix4d prevTb2m = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d prevTbc2bp = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d GpsTb2m = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d prevGpsTb2m = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d GpsTbc2bp = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d LidarTb2m = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d prevLidarTb2m = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d LidarTbc2bp = Eigen::Matrix4d::Identity();

        Eigen::Vector3d rlla  = Eigen::Vector3d::Zero(); 
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
        try {
            while (running) {
                auto data_frame = dataQueue.pop();
                if (!data_frame) {
                    if (!running) std::cout << "Data queue stopped, exiting factor thread.\n";
                    break;
                }
                pcl::PointCloud<pcl::PointXYZI>::Ptr points(new pcl::PointCloud<pcl::PointXYZI>());
                *points = std::move(data_frame->points.pointsBody);
                const Eigen::Vector3d& lla = data_frame->position.back().pose;
                const Eigen::Matrix3d& Cb2m = data_frame->position.back().orientation.toRotationMatrix().cast<double>();
                pcl::PointCloud<pcl::PointXYZI>::Ptr pointsMap(new pcl::PointCloud<pcl::PointXYZI>());
                if (is_first_keyframe){
                    rlla = lla;
                    Eigen::Vector3d tb2m = -registerCallback.lla2ned(lla.x(),lla.y(),lla.z(),rlla.x(),rlla.y(),rlla.z());
                    Eigen::Matrix4d Tb2m = Eigen::Matrix4d::Identity();
                    Tb2m.block<3,3>(0,0) = Cb2m.cast<double>();
                    Tb2m.block<3,1>(0,3) = tb2m;
                    pcl::transformPointCloud(*points,*pointsMap,Tb2m);
                    registerCallback.registration->setInputTarget(pointsMap);
                    prevTb2m = Tb2m;

                     // GTSAM part for the first keyframe
                    gtsam::Pose3 currGpsTb2m(Tb2m);

                    auto data_factor = std::make_unique<GtsamFactorData>();
                    data_factor->frame_id = data_frame->points.frame_id;
                    data_factor->pointsBody = std::move(points);
                    // Set initial estimates for the first state
                    data_factor->estimatedPoseFactor = currGpsTb2m;

                    // Add a strong GPS prior to anchor the graph
                    data_factor->has_gps_factor = true; 
                    data_factor->insFactor = currGpsTb2m;
                    gtsam::Vector6 anchor_sigmas;
                    anchor_sigmas << gtsam::Vector3::Constant(1e-4), gtsam::Vector3::Constant(1e-3);  
                    data_factor->insNoiseModel = gtsam::noiseModel::Diagonal::Sigmas(anchor_sigmas);
                    factorQueue.push(std::move(data_factor));
                    is_first_keyframe = false;
                    continue;
                }
                registerCallback.registration->setInputSource(points);
                Eigen::Matrix4d predTb2m = Eigen::Matrix4d::Identity();
                int ndt_iter = 0;
                Eigen::Matrix<double, 6, 6> lidarCov = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
                Eigen::Matrix4d estimatedTb2m = prevTb2m * prevTbc2bp;
                if (data_frame->position.back().poseStdDev.norm() > 1.0f) {
                    predTb2m = estimatedTb2m;
                    // std::cout << "Warning: GPS unreliable. Using motion model for NDT guess." << std::endl;
                }else {
                    Eigen::Vector3d tb2m = -registerCallback.lla2ned(lla.x(),lla.y(),lla.z(),rlla.x(),rlla.y(),rlla.z());
                    Eigen::Matrix4d Tb2m = Eigen::Matrix4d::Identity();
                    Tb2m.block<3,3>(0,0) = Cb2m.cast<double>();
                    Tb2m.block<3,1>(0,3) = tb2m;
                    predTb2m = Tb2m;
                    GpsTb2m = Tb2m;
                    GpsTbc2bp = prevGpsTb2m.inverse()*GpsTb2m;
                    prevTb2m = GpsTb2m;
                    prevTbc2bp = GpsTbc2bp;
                    prevGpsTb2m =GpsTb2m;
                    // std::cout << "Logging: GPS reliable. Using gps for NDT guess." << std::endl;
                }
                auto align_start = std::chrono::high_resolution_clock::now();
                registerCallback.registration->align(*pointsMap, predTb2m.cast<float>());
                auto align_end = std::chrono::high_resolution_clock::now();
                auto align_duration = std::chrono::duration_cast<std::chrono::milliseconds>(align_end - align_start);
                if (registerCallback.registration->hasConverged()) {
                    std::cout << "Registration converged." << std::endl;
                    LidarTb2m = registerCallback.registration->getFinalTransformation().cast<double>();
                    registerCallback.registration->setInputTarget(pointsMap);
                    prevTb2m = LidarTb2m;
                    LidarTbc2bp = prevLidarTb2m.inverse()*LidarTb2m;
                    prevTbc2bp = LidarTbc2bp;
                    prevLidarTb2m = LidarTb2m;
                    if (ndt_omp) {
                        auto ndt_result = ndt_omp->getResult();
                        ndt_iter = ndt_result.iteration_num;
                        const auto& hessian = ndt_result.hessian;
                        Eigen::Matrix<double, 6, 6> regularized_hessian = hessian + (Eigen::Matrix<double, 6, 6>::Identity() * 1e-6);
                        if (hessian.determinant() > 1e-6) {
                            lidarCov = -hessian.inverse();
                            std::cout << "Covariance estimated from NDT Hessian.\n";
                        } else {
                            std::cerr << "Hessian singular; using default covariance.\n";
                        }
                    }
                } 
                // else {
                //     std::cout << "Registration failed to converge." << std::endl;
                // }
                std::cout << "........................................" << std::endl;
                std::cout << "Factor Thread..........................." << std::endl;
                std::cout << "Frame ID................................" << data_frame->points.frame_id << std::endl;
                // std::cout << "Position stndrdDev..................." << data_frame->position.back().poseStdDev.norm() << std::endl;
                // std::cout << "Number points........................" << points->size() << std::endl;
                // std::cout << "Alignment Time......................." << align_duration.count() << " ms" << std::endl;
                // std::cout << "Number Iteration....................." << ndt_iter << std::endl;
                // std::cout << "tran source to target norm..........." << prevTbc2bp.block<3, 1>(0, 3).norm() << std::endl;
                // std::cout << "tran Ld source to target norm........" << LidarTbc2bp.block<3, 1>(0, 3).norm() << std::endl;
                // std::cout << "tran GPS source to target norm......." << GpsTbc2bp.block<3, 1>(0, 3).norm() << std::endl;
                // std::cout << "diff Aligned to Gps trans norm......." << LidarTbc2bp.block<3, 1>(0, 3).norm() - GpsTbc2bp.block<3, 1>(0, 3).norm() << std::endl;
                std::cout << "Tb2m....................................\n" << GpsTb2m << std::endl;
                // std::cout << "T L body to map......................\n" << LidarTb2m << std::endl;
                // std::cout << "6D Covariance........................\n" << lidarCov << std::endl;
                std::cout << "........................................." << std::endl;

                // 3. Create and Populate GtsamFactorData
                auto data_factor = std::make_unique<GtsamFactorData>();
                data_factor->frame_id = data_frame->points.frame_id;
                data_factor->pointsBody = std::move(points);
                gtsam::Pose3 estimatedPose(std::move(estimatedTb2m));
                data_factor->estimatedPoseFactor = std::move(estimatedPose);
                // 2. Populate LiDAR Factor Data
                data_factor->lidarFactor = gtsam::Pose3(prevTbc2bp);
                Eigen::Matrix<double, 6, 6> lidarNoiseModel = registerCallback.reorderCovarianceForGTSAM(std::move(lidarCov));
                data_factor->lidarNoiseModel = gtsam::noiseModel::Gaussian::Covariance(std::move(lidarNoiseModel));

                // 3. Populate GPS Factor Data
                if (data_frame->position.back().poseStdDev.norm() <= 1.0f) {
                    data_factor->has_gps_factor = true;
                    data_factor->insFactor = gtsam::Pose3(GpsTb2m);
                    const auto& currPoseStdDev = data_frame->position.back().poseStdDev;
                    data_factor->insNoiseModel = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, currPoseStdDev.x(), currPoseStdDev.y(), currPoseStdDev.z()).finished());
                }
                factorQueue.push(std::move(data_factor));   
            }
        } catch (const std::exception& e) {
            std::cerr << "Factor thread error: " << e.what() << "\n";
        }
        std::cout << "Factor thread exiting\n";
    });

    auto gtsam_thread = std::thread([&factorQueue]() {
        bool is_first_keyframe = true;
        gtsam::ISAM2Params isam2_params;
        isam2_params.relinearizeThreshold = 0.1;
        isam2_params.relinearizeSkip = 1;
        gtsam::ISAM2 isam2(isam2_params);
        try{
            while (running) {
                auto data_factor = factorQueue.pop();
                if (!data_factor) {
                    if (!running) std::cout << "Factor queue stopped, exiting Gtsam thread.\n";
                    break;
                }

                gtsam::NonlinearFactorGraph newFactors;
                gtsam::Values newEstimates;
                uint64_t id = data_factor->frame_id;

                // Add the initial estimate for the new pose
                newEstimates.insert(Symbol('x', id), data_factor->estimatedPoseFactor);

                if (is_first_keyframe) {
                    // Add a Prior for the first pose to anchor the graph
                    newFactors.add(gtsam::PriorFactor<gtsam::Pose3>(Symbol('x', id), data_factor->insFactor, data_factor->insNoiseModel));
                    is_first_keyframe = false;
                    continue;
                } 

                if (data_factor->has_gps_factor) {
                    newFactors.add(gtsam::PriorFactor<gtsam::Pose3>(Symbol('x', id), data_factor->insFactor, data_factor->insNoiseModel));
                }
                newFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(Symbol('x', id - 1), Symbol('x', id),data_factor->lidarFactor, data_factor->lidarNoiseModel));
                
                isam2.update(newFactors, newEstimates);

                // Periodically print a more detailed summary
                gtsam::Values Val = isam2.calculateEstimate();
                gtsam::Pose3 Tb2m = Val.at<gtsam::Pose3>(Symbol('x', id));

                std::cout << "........................................" << std::endl;
                std::cout << "Gtsam Thread............................" << std::endl;
                std::cout << "Frame ID................................" << id << std::endl;
                std::cout << "New factors added this step............." << newFactors.size() << std::endl;
                std::cout << "Total factors in graph.................." << isam2.size() << std::endl;
                std::cout << "Optimized Tb2m..........................\n" << Tb2m.matrix() << std::endl;
                std::cout << "........................................" << std::endl;
                

            }
        } catch (const std::exception& e) {
            std::cerr << "Gtsam thread error: " << e.what() << "\n";
        }
        std::cout << "Gtsam thread exiting\n";
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
    factorQueue.stop();

    if (lidar_iothread.joinable()) lidar_iothread.join();
    if (comp_iothread.joinable()) comp_iothread.join();
    if (sync_thread.joinable()) sync_thread.join();
    if (factor_thread.joinable()) factor_thread.join();
    if (gtsam_thread.joinable()) gtsam_thread.join();
    
    std::cout << "All threads have been joined. Shutdown complete." << std::endl;
}