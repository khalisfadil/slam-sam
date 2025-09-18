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
    auto factor_thread = std::thread([&registerCallback, &dataQueue]() {
        bool is_first_keyframe = true;
        Eigen::Matrix4d prevTb2m = Eigen::Matrix4f::Identity();
        Eigen::Matrix4d prevTbc2bp = Eigen::Matrix4f::Identity();
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
                    is_first_keyframe = false;
                    continue;
                }
                registerCallback.registration->setInputSource(points);
                Eigen::Matrix4d predTb2m = Eigen::Matrix4d::Identity();
                int iter = 0;
                Eigen::Matrix<double, 6, 6> lidar_factor_cov = Eigen::Matrix<double, 6, 6>::Identity() * 0.01;
                if (data_frame->position.back().poseStdDev.norm() > 5.0f) {
                    predTb2m = prevTb2m * prevTbc2bp;
                    std::cout << "Warning: GPS unreliable. Using motion model for NDT guess." << std::endl;
                }else {
                    Eigen::Vector3d tb2m = -registerCallback.lla2ned(lla.x(),lla.y(),lla.z(),rlla.x(),rlla.y(),rlla.z());
                    Eigen::Matrix4d Tb2m = Eigen::Matrix4d::Identity();
                    Tb2m.block<3,3>(0,0) = Cb2m.cast<double>();
                    Tb2m.block<3,1>(0,3) = tb2m;
                    predTb2m = Tb2m;
                }
                auto align_start = std::chrono::high_resolution_clock::now();
                registerCallback.registration->align(*pointsMap, predTb2m.cast<float>());
                auto align_end = std::chrono::high_resolution_clock::now();
                auto align_duration = std::chrono::duration_cast<std::chrono::milliseconds>(align_end - align_start);
                if (registerCallback.registration->hasConverged()) {
                    Eigen::Matrix4d Tbc2bp = registerCallback.registration->getFinalTransformation().cast<double>();
                    registerCallback.registration->setInputTarget(pointsMap);
                    prevTbc2bp = Tbc2bp;
                    prevTb2m = predTb2m;
                    if (ndt_omp) {
                        auto ndt_result = ndt_omp->getResult();
                        iter = ndt_result.iteration_num;
                        const auto& hessian = ndt_result.hessian;
                        if (hessian.determinant() > 1e-6) {
                            lidar_factor_cov = -hessian.inverse();
                            std::cout << "Covariance estimated from NDT Hessian.\n";
                        } else {
                            std::cerr << "Hessian singular; using default covariance.\n";
                        }
                    }
                } else {
                    std::cout << "Registration failed to converge." << std::endl;
                }
                std::cout << "----------------------------------------" << std::endl;
                std::cout << "Number points.................." << data_frame->points.pointsBody.size() << std::endl;
                std::cout << "Alignment Time................." << align_duration.count() << " ms" << std::endl;
                std::cout << "Number Iteration..............." << iter << std::endl;
                std::cout << "tran source to target norm....." << prevTbc2bp.block<3, 1>(0, 3).norm() << std::endl;
                std::cout << "T source to target.............\n" << prevTb2m << std::endl;
                std::cout << "6D Covariance..................\n" << lidar_factor_cov << std::endl;
                std::cout << "----------------------------------------" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Factor thread error: " << e.what() << "\n";
        }
        std::cout << "Factor thread exiting\n";
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

    if (lidar_iothread.joinable()) lidar_iothread.join();
    if (comp_iothread.joinable()) comp_iothread.join();
    if (sync_thread.joinable()) sync_thread.join();
    if (factor_thread.joinable()) factor_thread.join();
    
}