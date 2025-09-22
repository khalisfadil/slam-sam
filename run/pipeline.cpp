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
    auto gtsam_thread = std::thread([&registerCallback, &dataQueue, &vizQueue]() {
        // --- MAPS AND NDT ARE NOW DECLARED AND OWNED BY THIS THREAD ---
        const double VOXEL_SIZE = 5.0; 
        const double LOOP_CLOSURE_TIME_THRESHOLD = 180.0;
        const int NEIGHBOR_SEARCH_SIZE = 1;

        PointsHashMap pointsArchive;
        VoxelHashMap spatialArchive;

        Eigen::Vector3d rlla  = Eigen::Vector3d::Zero(); 
        Eigen::Matrix4d lidarFactorSourceTb2m = Eigen::Matrix4d::Identity();

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

                pcl::PointCloud<pcl::PointXYZI>::Ptr pointsBody(new pcl::PointCloud<pcl::PointXYZI>());
                *pointsBody = std::move(data_frame->points.pointsBody);
                const Eigen::Vector3d& lla = data_frame->position.back().pose;
                const Eigen::Matrix3d& Cb2m = data_frame->position.back().orientation.toRotationMatrix().cast<double>();
                Eigen::Matrix4d Tb2m = Eigen::Matrix4d::Identity();
                

                uint64_t id = data_frame->points.frame_id;
                double timestamp = data_frame->timestamp;
                int ndt_iter = 0;
                std::chrono::milliseconds align_duration;

                Eigen::Matrix<double, 6, 6> lidarCov = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
                Eigen::Matrix<double, 6, 6> loopCov = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
                Eigen::Vector<double, 6> lidarStdDev = Eigen::Vector<double, 6>::Zero();
                Eigen::Vector<double, 6> insStdDev = Eigen::Vector<double, 6>::Zero();

                gtsam::NonlinearFactorGraph newFactors;
                gtsam::Values newEstimates;

                if (is_first_keyframe) {
                    rlla = lla;
                    Eigen::Vector3d tb2m = -registerCallback.lla2ned(lla.x(),lla.y(),lla.z(),rlla.x(),rlla.y(),rlla.z());
                    Tb2m = Eigen::Matrix4d::Identity();
                    Tb2m.block<3,3>(0,0) = Cb2m.cast<double>();
                    Tb2m.block<3,1>(0,3) = tb2m;

                    gtsam::Pose3 insFactor(Tb2m);
                    gtsam::Vector6 insNoise;
                    insNoise << gtsam::Vector3::Constant(1e-4), gtsam::Vector3::Constant(1e-3);
                    gtsam::SharedNoiseModel insNoiseModel = gtsam::noiseModel::Diagonal::Sigmas(std::move(insNoise));
                    newEstimates.insert(Symbol('x', id), insFactor);
                    newFactors.add(gtsam::PriorFactor<gtsam::Pose3>(Symbol('x', id), std::move(insFactor), std::move(insNoiseModel)));
                } else {
                    Eigen::Vector3d tb2m = -registerCallback.lla2ned(lla.x(),lla.y(),lla.z(),rlla.x(),rlla.y(),rlla.z());
                    Tb2m = Eigen::Matrix4d::Identity();
                    Tb2m.block<3,3>(0,0) = Cb2m.cast<double>();
                    Tb2m.block<3,1>(0,3) = tb2m;

                    gtsam::Pose3 initialFactor(lidarFactorSourceTb2m);
                    newEstimates.insert(Symbol('x', id), initialFactor);
                    pcl::PointCloud<pcl::PointXYZI>::Ptr lidarFactorPointsSource(new pcl::PointCloud<pcl::PointXYZI>());
                    pcl::PointCloud<pcl::PointXYZI>::Ptr lidarFactorPointsTarget(new pcl::PointCloud<pcl::PointXYZI>());
                    const auto& lidarFactorPointsArchive = pointsArchive.at(id - 1);
                    gtsam::Pose3 lidarFactorTargetTb2m = Val.at<gtsam::Pose3>(Symbol('x', id - 1));
                    pcl::transformPointCloud(*lidarFactorPointsArchive.points, *lidarFactorPointsTarget, lidarFactorTargetTb2m.matrix());
                    registerCallback.registration->setInputTarget(lidarFactorPointsTarget);
                    registerCallback.registration->setInputSource(pointsBody);
                    auto align_start = std::chrono::high_resolution_clock::now();
                    registerCallback.registration->align(*lidarFactorPointsSource, lidarFactorSourceTb2m.cast<float>());
                    auto align_end = std::chrono::high_resolution_clock::now();
                    align_duration = std::chrono::duration_cast<std::chrono::milliseconds>(align_end - align_start);
                    if (registerCallback.registration->hasConverged()) {
                        std::cout << "Registration converged." << std::endl;
                        lidarFactorSourceTb2m = registerCallback.registration->getFinalTransformation().cast<double>();
                        Eigen::Matrix4d lidarTbs2bt = lidarFactorTargetTb2m.matrix().inverse()*lidarFactorSourceTb2m;
                        if (ndt_omp) {
                            auto ndt_result = ndt_omp->getResult();
                            ndt_iter = ndt_result.iteration_num;
                            const auto& hessian = ndt_result.hessian;
                            Eigen::Matrix<double, 6, 6> regularized_hessian = hessian + (Eigen::Matrix<double, 6, 6>::Identity() * 1e-6);
                            if (regularized_hessian.determinant() > 1e-6) {
                                lidarCov = -regularized_hessian.inverse()*10;
                                lidarStdDev = lidarCov.diagonal().cwiseSqrt();
                                std::cout << "Covariance estimated from NDT Hessian.\n";
                            }
                        } 
                        gtsam::Pose3 lidarFactor = gtsam::Pose3(std::move(lidarTbs2bt));
                        gtsam::SharedNoiseModel lidarNoiseModel = gtsam::noiseModel::Gaussian::Covariance(registerCallback.reorderCovarianceForGTSAM(std::move(lidarCov)));
                        newFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(Symbol('x', id - 1), Symbol('x', id), std::move(lidarFactor), std::move(lidarNoiseModel)));
                    } else {
                        Eigen::Matrix4d lidarTbs2bt = lidarFactorTargetTb2m.matrix().inverse()*lidarFactorSourceTb2m;
                        gtsam::Pose3 lidarFactor = gtsam::Pose3(std::move(lidarTbs2bt));
                        lidarCov = Eigen::Matrix<double, 6, 6>::Identity() * 1.0;
                        lidarStdDev = lidarCov.diagonal().cwiseSqrt();
                        gtsam::SharedNoiseModel lidarNoiseModel = gtsam::noiseModel::Gaussian::Covariance(registerCallback.reorderCovarianceForGTSAM(std::move(lidarCov)));
                        newFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(Symbol('x', id - 1), Symbol('x', id), std::move(lidarFactor), std::move(lidarNoiseModel)));
                    }
                    // Also add a GPS prior if the data is reliable.
                    if (data_frame->position.back().poseStdDev.norm() < 0.1f) {
                        gtsam::Pose3 insFactor(Tb2m);
                        const auto& insFactorStdDev = data_frame->position.back().poseStdDev;
                        insStdDev << insFactorStdDev.x(), insFactorStdDev.y(), insFactorStdDev.z(),0.01, 0.01, 0.01;
                        gtsam::SharedNoiseModel insNoiseModel = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.01, 0.01, 0.01, insFactorStdDev.x(), insFactorStdDev.y(), insFactorStdDev.z()).finished());
                        newFactors.add(gtsam::PriorFactor<gtsam::Pose3>(Symbol('x', id), std::move(insFactor), std::move(insNoiseModel)));
                    } else {
                        insStdDev = Eigen::Vector<double, 6>::Ones() * 10;

                    }
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
                    gtsam::Pose3 prevTb2m = Val.at<gtsam::Pose3>(Symbol('x', id -1));
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
                std::cout << std::fixed << "Ins Std Dev (m, rad)....................\n" << insStdDev.transpose() << std::endl;
                std::cout << std::fixed << "Lidar Std Dev (m, rad)..................\n" << lidarStdDev.transpose() << std::endl;
                std::cout << std::fixed << "New factors added this step............." << newFactors.size() << std::endl;
                std::cout << std::fixed << "Total factors in graph.................." << isam2.size() << std::endl;
                std::cout << std::fixed << "Tb2m....................................\n" << Tb2m << std::endl;
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
        viewer->setCameraPosition(0, 0, -50, 0, 0, 0, 1, 0, 0); // This initial position is no longer essential

        // VoxelGrid filter...
        pcl::VoxelGrid<pcl::PointXYZI> vg;
        vg.setLeafSize(1.5f, 1.5f, 1.5f);

        pcl::PointCloud<pcl::PointXYZI>::Ptr map_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr trajectory_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

         bool pose_viz_added = false;

        while (running && !viewer->wasStopped()) {
            auto vizData = vizQueue.pop();
            if (!vizData) {
                if (!running) std::cout << "Visualization queue stopped, exiting viz thread.\n";
                break;
            }

            map_cloud->clear();
            trajectory_cloud->clear();

            // ADD THESE TWO VARIABLES to track the latest pose
            gtsam::Pose3 latest_pose;
            uint64_t max_id = 0;

            for (const auto& key_value : *(vizData->poses)) {
                uint64_t frame_id = gtsam::Symbol(key_value.key).index();
                
                auto it = vizData->points->find(frame_id);
                if (it != vizData->points->end()) {
                    auto& keypoint = it->second;
                    gtsam::Pose3 pose = key_value.value.cast<gtsam::Pose3>();
                    
                    // --- Keep track of the latest pose ---
                    if (frame_id > max_id) {
                        max_id = frame_id;
                        latest_pose = pose;
                    }
                    
                    // (The rest of your cloud construction logic is unchanged)
                    pcl::transformPointCloud(*keypoint.points, *transformed_cloud, pose.matrix().cast<float>());
                    *map_cloud += *transformed_cloud;

                    pcl::PointXYZRGB trajectory_point;
                    trajectory_point.x = pose.translation().x();
                    trajectory_point.y = pose.translation().y();
                    trajectory_point.z = pose.translation().z();
                    trajectory_point.r = 255;
                    trajectory_point.g = 10;
                    trajectory_point.b = 10;
                    trajectory_cloud->push_back(trajectory_point);
                }
            }

            // Downsample the map (unchanged)
            pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_map(new pcl::PointCloud<pcl::PointXYZI>());
            if (!map_cloud->empty()) {
                vg.setInputCloud(map_cloud);
                vg.filter(*downsampled_map);
            }

            // Update visualizer point clouds (unchanged)
            pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color_handler(downsampled_map, "intensity");
            if (!viewer->updatePointCloud(downsampled_map, color_handler, "map_cloud")) {
                viewer->addPointCloud(downsampled_map, color_handler, "map_cloud");
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "map_cloud");
            }
            // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color_handler(transformed_cloud, "intensity");
            // if (!viewer->updatePointCloud(transformed_cloud, color_handler, "transformed_cloud")) {
            //     viewer->addPointCloud(transformed_cloud, color_handler, "transformed_cloud");
            //     viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "transformed_cloud");
            // }
            if (!viewer->updatePointCloud(trajectory_cloud, "trajectory_cloud")) {
                viewer->addPointCloud(trajectory_cloud, "trajectory_cloud");
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "trajectory_cloud");
            }

            // --- ADD THIS BLOCK TO UPDATE THE CAMERA POSITION ---
            // After updating the clouds, move the camera to follow the latest pose
            // if (max_id > 0) { // Ensure we have a valid latest pose
            //     const auto& current_pos = latest_pose.translation();
            //     viewer->setCameraPosition(
            //         current_pos.x(), current_pos.y(), -80, // Camera position: behind and above the vehicle
            //         current_pos.x(), current_pos.y(), current_pos.z(),               // Viewpoint: where the vehicle is
            //         1, 0, 0                                                          // Up vector: Z-axis is up
            //     );

            //     // 2. Convert gtsam::Pose3 to Eigen::Affine3f for the visualizer
            //     Eigen::Affine3f vehicle_pose_affine = Eigen::Affine3f::Identity();
            //     vehicle_pose_affine.matrix() = latest_pose.matrix().cast<float>();

            //     // 3. Add or update the vehicle's coordinate system
            //     if (!pose_viz_added) {
            //         // Add the coordinate system with a scale of 2.0 and a unique ID
            //         viewer->addCoordinateSystem(2.0, vehicle_pose_affine, "vehicle_pose");
            //         pose_viz_added = true;
            //     } else {
            //         // Update the pose of the existing coordinate system
            //         viewer->updateCoordinateSystemPose("vehicle_pose", vehicle_pose_affine);
            //     }
            // }

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
    
    std::cout << "All threads have been joined. Shutdown complete." << std::endl;
}