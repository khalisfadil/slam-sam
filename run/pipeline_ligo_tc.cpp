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

                const double& start_interval = *keyLidarTs; //
                const double& end_interval = max_lidar_time; //

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
                        dataFrame->ins.push_back(data);
                    }
                }

                // Add interpolated end point
                CompFrame end_frame = getInterpolated(end_interval, current_comp_window);
                dataFrame->ins.push_back(end_frame); //

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
    auto gtsam_thread = std::thread([&registerCallback, &compCallback, &dataQueue, &vizQueue]() {

        // =================================================================================
        // A. PARAMETERS & CONSTANTS
        // =================================================================================
        const uint8_t MIN_INS_FIX_STATUS = 3; // Require a good fix (e.g., RTK Float/Fixed) for a quality start
        const uint8_t MIN_GNSS_FIX_STATUS = 3; // Can accept a 3D fix for continuous updates

        const gtsam::Vector3 STATIC_BIAS_ACCELEROMETER = compCallback.getStaticBiasAccelerometer();
        const gtsam::Vector3 STATIC_BIAS_GYROSCOPE = compCallback.getStaticBiasGyroscope();
        const gtsam::Vector3 VELOCITY_RANDOM_WALK = compCallback.getVelocityRandomWalk();
        const gtsam::Vector3 ANGULAR_RANDOM_WALK = compCallback.getAngularRandomWalk();
        const gtsam::Vector3 BIAS_RANDOM_WALK_ACC = compCallback.getBiasRandomWalkAccelerometer();
        const gtsam::Vector3 BIAS_RANDOM_WALK_GYRO = compCallback.getBiasRandomWalkGyroscope();
        const gtsam::Vector3 BIAS_INSTABILITY_ACC = compCallback.getBiasInstabilityAccelerometer();
        const gtsam::Vector3 BIAS_INSTABILITY_GYRO = compCallback.getBiasInstabilityGyroscope();

        // =================================================================================
        // B. NDT SETUP
        // =================================================================================
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
        } 

        // =================================================================================
        // C. STATE & ARCHIVE VARIABLES
        // =================================================================================
        StateHashMap insStateArchive;
        PointsHashMap pointsArchive;
        Eigen::Vector3d ins_rlla = Eigen::Vector3d::Zero();
        bool is_first_keyframe = true;
        uint64_t last_id = 0;
        double last_timestamp = 0.0;
        
        gtsam::NavState prev_state_optimized;
        gtsam::imuBias::ConstantBias prev_bias_optimized;
        std::shared_ptr<gtsam::PreintegrationCombinedParams> imu_params;
        std::shared_ptr<gtsam::PreintegratedCombinedMeasurements> imu_preintegrator;

        // =================================================================================
        // D. GTSAM SETUP
        // =================================================================================
        gtsam::ISAM2Params isam2_params;
        isam2_params.relinearizeThreshold = 0.1;
        isam2_params.relinearizeSkip = 1;
        gtsam::ISAM2 isam2(isam2_params);
        gtsam::Values currentEstimates;

        // =================================================================================
        // E. MAIN THREAD LOOP
        // =================================================================================
        try {
            while (running) {
                auto data_frame = dataQueue.pop();
                if (!data_frame) {
                    if (!running) std::cout << "Factor queue stopped, exiting Gtsam thread.\n";
                    break;
                }
                // --- 1. Data Extraction & Validation -- 
                pcl::PointCloud<pcl::PointXYZI>::Ptr pointsBody(new pcl::PointCloud<pcl::PointXYZI>());
                *pointsBody = std::move(data_frame->points.pointsBody);
                const auto& ins = data_frame->ins.back();
                const uint64_t& id = data_frame->points.frame_id;
                const auto& timestamp = data_frame->timestamp;
                const bool& insValid = (ins.GNSSFixStatus_20 > MIN_INS_FIX_STATUS || ins.timestamp_20 > 0.0 || ins.latitude_20 != 0.0 || ins.longitude_20 != 0.0);         // this check because reading is scalar but it give 0 rad.
                const bool& gnssValid = (ins.GNSSFixStatus_29 > MIN_GNSS_FIX_STATUS || ins.timestamp_29 > 0.0 || ins.latitude_29 != 0.0 || ins.longitude_29 != 0.0);        // this check because reading is scalar but it give 0 rad.                                                                                         // add better condition for this check
                const bool& lidarValid = (data_frame->points.pointsBody.size() > 0);   
                
                gtsam::NonlinearFactorGraph newFactors;
                gtsam::Values newEstimates;
                const Eigen::Vector3d ins_lla{ins.latitude_20, ins.longitude_20, ins.altitude_20};
                Eigen::Quaternionf ins_quat{ins.qw_20, ins.qx_20, ins.qy_20, ins.qz_20};
                const Eigen::Matrix3d Cb2m = ins_quat.toRotationMatrix().cast<double>();
                const gtsam::Rot3 ins_Cb2m{Cb2m};
                const gtsam::Vector3 ins_vNED{ins.velocityNorth_20, ins.velocityEast_20, ins.velocityDown_20};
                gtsam::NavState current_ins_state{};
                int ndt_iter = 0;
                // --- 2. Initialization (First Keyframe) ---
                if (is_first_keyframe) {
                    if (insValid) {
                        ins_rlla = ins_lla;
                        const gtsam::Point3 ins_tb2m{registerCallback.lla2ned(ins_lla.x(), ins_lla.y(), ins_lla.z(), ins_rlla.x(), ins_rlla.y(), ins_rlla.z())};
                        current_ins_state = gtsam::NavState{ins_Cb2m, ins_tb2m, ins_vNED};
                        gtsam::Vector3 ins_gravity(0.0, 0.0, compCallback.GravityWGS84(ins_lla.x(), ins_lla.y(), ins_lla.z()));
                        imu_params = std::make_shared<gtsam::PreintegrationCombinedParams>(ins_gravity);
                        gtsam::Vector3 accel_variances              = VELOCITY_RANDOM_WALK.array().square();
                        gtsam::Vector3 gyro_variances               = ANGULAR_RANDOM_WALK.array().square();
                        gtsam::Vector3 bias_accel_variances         = BIAS_RANDOM_WALK_ACC.array().square();
                        gtsam::Vector3 bias_gyro_variances          = BIAS_RANDOM_WALK_GYRO.array().square();
                        imu_params->accelerometerCovariance         = accel_variances.asDiagonal();
                        imu_params->gyroscopeCovariance             = gyro_variances.asDiagonal();
                        imu_params->biasAccCovariance               = bias_accel_variances.asDiagonal();
                        imu_params->biasOmegaCovariance             = bias_gyro_variances.asDiagonal();
                        imu_params->integrationCovariance           = gtsam::Matrix33::Identity() * 1e-8;
                        imu_params->biasAccOmegaInt                 = gtsam::Matrix66::Zero();
                        const auto& insInitialPose = current_ins_state.pose();
                        const auto& insInitialVelocity = current_ins_state.velocity();
                        const gtsam::imuBias::ConstantBias imuInitialBias(STATIC_BIAS_ACCELEROMETER, STATIC_BIAS_GYROSCOPE);
                        newEstimates.insert(gtsam::Symbol('x', id), insInitialPose);                                                                                                                                                                                                    
                        newEstimates.insert(gtsam::Symbol('v', id), insInitialVelocity);
                        newEstimates.insert(gtsam::Symbol('b', id), imuInitialBias); 
                        auto insPoseNoiseModel = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << ins.sigmaRoll_26, ins.sigmaPitch_26, ins.sigmaYaw_26, ins.sigmaLatitude_20, ins.sigmaLongitude_20, ins.sigmaAltitude_20).finished());
                        newFactors.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', id), insInitialPose, insPoseNoiseModel));
                        auto insVelocityNoiseModel = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << ins.sigmaVelocityNorth_25, ins.sigmaVelocityEast_25, ins.sigmaVelocityDown_25).finished());
                        newFactors.add(gtsam::PriorFactor<gtsam::Vector3>(gtsam::Symbol('v', id), insInitialVelocity, insVelocityNoiseModel));
                        gtsam::Vector6 bias_instability_sigmas;
                        bias_instability_sigmas << BIAS_INSTABILITY_ACC, BIAS_INSTABILITY_GYRO;
                        auto insBiasNoiseModel = gtsam::noiseModel::Diagonal::Sigmas(bias_instability_sigmas);
                        newFactors.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', id), imuInitialBias, insBiasNoiseModel));

                        isam2.update(newFactors, newEstimates);
                        currentEstimates = isam2.calculateEstimate();
                        gtsam::Pose3 prev_pose_optimized   = currentEstimates.at<gtsam::Pose3>(gtsam::Symbol('x', id));
                        gtsam::Vector3 prev_velocity_optimized = currentEstimates.at<gtsam::Vector3>(gtsam::Symbol('v', id));
                        prev_bias_optimized = currentEstimates.at<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', id));
                        prev_state_optimized = gtsam::NavState(prev_pose_optimized, prev_velocity_optimized);
                        imu_preintegrator = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(imu_params, prev_bias_optimized);

                        pointsArchive[id] = {pointsBody, timestamp};
                        insStateArchive[id] = {current_ins_state, timestamp};
                        last_id = id;
                        is_first_keyframe = false;

                        //3.5 push vizual
                        if (!currentEstimates.empty()) {
                            auto vizData = std::make_unique<VisualizationData>();
                            vizData->poses = std::make_shared<gtsam::Values>(currentEstimates);
                            vizData->points = std::make_shared<PointsHashMap>(pointsArchive);
                            vizData->insposes = std::make_shared<StateHashMap>(insStateArchive);
                            vizQueue.push(std::move(vizData));
                        }
                    }
                } else { // --- 3. Main Loop for Subsequent Keyframes ---
                    const gtsam::Point3 ins_tb2m{registerCallback.lla2ned(ins_lla.x(), ins_lla.y(), ins_lla.z(), ins_rlla.x(), ins_rlla.y(), ins_rlla.z())};
                    current_ins_state = gtsam::NavState{ins_Cb2m, ins_tb2m, ins_vNED};

                    // 3.1. Integrate IMU measurements between the last keyframe and this one.
                    imu_preintegrator->resetIntegrationAndSetBias(prev_bias_optimized);
                    for (const auto& measurement : data_frame->ins) {
                        // This assumes your IMU measurements are ordered correctly in the vector
                        double dt = measurement.timestamp_20 - last_timestamp;
                        if (dt > 0) {
                            gtsam::Vector3 accel(measurement.accelX_28, measurement.accelY_28, measurement.accelZ_28);
                            gtsam::Vector3 gyro(measurement.gyroX_28, measurement.gyroY_28, measurement.gyroZ_28);
                            imu_preintegrator->integrateMeasurement(accel, gyro, dt);
                        }
                        last_timestamp = measurement.timestamp_20;
                    }

                    // 3.2. PREDICT the current state using the IMU preintegration.
                    // This provides a high-quality initial guess for the new variables.
                    gtsam::NavState predicted_state = imu_preintegrator->predict(prev_state_optimized, prev_bias_optimized);
                    newEstimates.insert(gtsam::Symbol('x', id), predicted_state.pose());
                    newEstimates.insert(gtsam::Symbol('v', id), predicted_state.velocity());
                    newEstimates.insert(gtsam::Symbol('b', id), prev_bias_optimized); // Assume bias is constant for the initial guess
                    
                    // 3.3. Add the IMU factor to the graph.
                    newFactors.add(gtsam::CombinedImuFactor(
                        gtsam::Symbol('x', last_id), gtsam::Symbol('v', last_id),
                        gtsam::Symbol('x', id), gtsam::Symbol('v', id),
                        gtsam::Symbol('b', last_id), gtsam::Symbol('b', id),
                        *imu_preintegrator));

                    // 3.4. (Conceptual) Add Lidar Odometry Factor
                    if (lidarValid){
                        const auto& prev_points = pointsArchive.at(last_id).points;
                        ndt_omp->setInputSource(pointsBody);
                        ndt_omp->setInputTarget(prev_points);
                        gtsam::Pose3 imu_predicted_odometry = prev_state_optimized.pose().between(predicted_state.pose());
                        pcl::PointCloud<pcl::PointXYZI>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZI>());
                        ndt_omp->align(*aligned_cloud, imu_predicted_odometry.matrix().cast<float>());
                        gtsam::Pose3 lidar_odometry(ndt_omp->getFinalTransformation().cast<double>());
                        auto ndt_result = ndt_omp->getResult();
                        ndt_iter = ndt_result.iteration_num;
                        const auto& hessian = ndt_result.hessian;
                        Eigen::Matrix<double, 6, 6> regularized_hessian = hessian + (Eigen::Matrix<double, 6, 6>::Identity() * 1e-6);
                        Eigen::Matrix<double, 6, 6> lidar_cov = (-regularized_hessian).inverse();
                        gtsam::SharedNoiseModel lidar_noise_model = gtsam::noiseModel::Gaussian::Covariance(registerCallback.reorderCovarianceForGTSAM(lidar_cov));
                        newFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol('x', last_id), gtsam::Symbol('x', id), lidar_odometry, lidar_noise_model));
                    }
                    // 3.4. (Conceptual) Add GPS Factor
                    if(gnssValid){
                        const Eigen::Vector3d gnss_lla{ins.latitude_29, ins.longitude_29, ins.altitude_29};
                        const gtsam::Point3 gnss_tb2m{registerCallback.lla2ned(gnss_lla.x(), gnss_lla.y(), gnss_lla.z(), ins_rlla.x(), ins_rlla.y(), ins_rlla.z())};
                        auto gnssNoiseModel = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << ins.sigmaLatitude_29, ins.sigmaLongitude_29, ins.sigmaAltitude_29).finished());
                        newFactors.add(gtsam::GPSFactor(gtsam::Symbol('x', id), gnss_tb2m, gnssNoiseModel));
                    }

                    isam2.update(newFactors, newEstimates);
                    currentEstimates = isam2.calculateEstimate();
                    gtsam::Pose3 prev_pose_optimized   = currentEstimates.at<gtsam::Pose3>(gtsam::Symbol('x', id));
                    gtsam::Vector3 prev_velocity_optimized = currentEstimates.at<gtsam::Vector3>(gtsam::Symbol('v', id));
                    prev_bias_optimized = currentEstimates.at<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', id));
                    prev_state_optimized = gtsam::NavState(prev_pose_optimized, prev_velocity_optimized);

                    // =========================================================================
                    // --- NEW: POSE COMPARISON OUTPUT ---
                    // =========================================================================
                    std::cout << "\n--- POSE COMPARISON (Frame ID: " << id << ") ---\n";
                    std::cout << "Raw INS Pose:     T = [" << current_ins_state.pose().translation().transpose() << "]\n";
                    std::cout << "GTSAM Opt. Pose:  T = [" << prev_pose_optimized.translation().transpose() << "]\n";
                    // =========================================================================

                    pointsArchive[id] = {pointsBody, timestamp};
                    insStateArchive[id] = {current_ins_state, timestamp};
                    last_id = id;   

                    //3.5 push vizual
                    if (!currentEstimates.empty()) {
                        auto vizData = std::make_unique<VisualizationData>();
                        vizData->poses = std::make_shared<gtsam::Values>(currentEstimates);
                        vizData->points = std::make_shared<PointsHashMap>(pointsArchive);
                        vizData->insposes = std::make_shared<StateHashMap>(insStateArchive);
                        vizQueue.push(std::move(vizData));
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Gtsam thread error: " << e.what() << "\n";
        }
        std::cout << "Gtsam thread exiting\n";
    });
    //####################################################################################################
    auto viz_thread = std::thread([&vizQueue]() { // Added &running to the capture list
        auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("GTSAM Optimized Map");
        viewer->setBackgroundColor(0.1, 0.1, 0.1);
        viewer->addCoordinateSystem(10.0, "world_origin");
        viewer->initCameraParameters();

        // --- Camera Smoothing Parameters ---
        const double kSmoothingFactor = 0.1;
        const Eigen::Vector3d kCameraOffset(0.0, 0.0, -250.0);
        const Eigen::Vector3d kUpVector(1.0, 0.0, 0.0); // Z-down, X-up view
        Eigen::Vector3d target_focal_point(0.0, 0.0, 0.0);
        Eigen::Vector3d current_focal_point = target_focal_point;
        Eigen::Vector3d current_cam_pos = target_focal_point + kCameraOffset;

        // --- Visualization State ---
        const size_t kSlidingWindowSize = 2;
        std::set<uint64_t> displayed_frame_ids;
        pcl::VoxelGrid<pcl::PointXYZI> vg;
        vg.setLeafSize(0.5f, 0.5f, 0.5f);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr trajectory_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ins_trajectory_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        while (running && !viewer->wasStopped()) {
            auto vizData = vizQueue.pop();
            if (!vizData) {
                if (!running) std::cout << "Visualization queue stopped, exiting viz thread.\n";
                break;
            }

            if (vizData->poses->empty()) {
                viewer->spinOnce(100);
                continue;
            }

            // --- 1. GET THE SET OF VALID FRAME IDS TO DISPLAY ---
            std::vector<uint64_t> valid_ids;
            for (const auto& key_value : *(vizData->poses)) {
                gtsam::Symbol symbol(key_value.key);
                if (symbol.chr() == 'x') {
                    uint64_t id = symbol.index();
                    if (vizData->points->count(id)) {
                        valid_ids.push_back(id);
                    }
                }
            }

            std::set<uint64_t> desired_ids;
            if (!valid_ids.empty()) {
                size_t start_index = (valid_ids.size() > kSlidingWindowSize) ? (valid_ids.size() - kSlidingWindowSize) : 0;
                for (size_t i = start_index; i < valid_ids.size(); ++i) {
                    desired_ids.insert(valid_ids[i]);
                }
            }

            // --- 2. UPDATE POINT CLOUDS (SLIDING WINDOW) ---
            std::vector<uint64_t> ids_to_remove;
            for (uint64_t displayed_id : displayed_frame_ids) {
                if (desired_ids.find(displayed_id) == desired_ids.end()) {
                    ids_to_remove.push_back(displayed_id);
                }
            }
            for (uint64_t id : ids_to_remove) {
                viewer->removePointCloud("map_cloud_" + std::to_string(id));
                displayed_frame_ids.erase(id);
            }

            for (uint64_t id : desired_ids) {
                const auto& raw_cloud = vizData->points->at(id).points;
                gtsam::Pose3 optimized_pose = vizData->poses->at<gtsam::Pose3>(gtsam::Symbol('x', id));
                pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZI>());
                pcl::transformPointCloud(*raw_cloud, *transformed_cloud, optimized_pose.matrix().cast<float>());
                vg.setInputCloud(transformed_cloud);
                vg.filter(*downsampled_cloud);
                std::string cloud_id = "map_cloud_" + std::to_string(id);
                pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color_handler(downsampled_cloud, "intensity");
                if (displayed_frame_ids.count(id)) {
                    viewer->updatePointCloud(downsampled_cloud, color_handler, cloud_id);
                } else {
                    viewer->addPointCloud(downsampled_cloud, color_handler, cloud_id);
                    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloud_id);
                    displayed_frame_ids.insert(id);
                }
            }

            // --- 3. UPDATE TRAJECTORIES ---
            // Optimized GTSAM Trajectory (Deep Pink)
            trajectory_cloud->clear();
            for (const auto& key_value : *(vizData->poses)) {
                gtsam::Symbol symbol(key_value.key);
                if (symbol.chr() == 'x') { // Process only pose variables
                    gtsam::Pose3 pose = key_value.value.cast<gtsam::Pose3>();
                    pcl::PointXYZRGB p;
                    p.x = pose.translation().x(); p.y = pose.translation().y(); p.z = pose.translation().z();
                    p.r = 255; p.g = 20; p.b = 147;
                    trajectory_cloud->push_back(p);
                }
            }
            if (!viewer->updatePointCloud(trajectory_cloud, "trajectory_cloud")) {
                viewer->addPointCloud(trajectory_cloud, "trajectory_cloud");
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "trajectory_cloud");
            }

            // Raw INS Trajectory (Deep Sky Blue)
            ins_trajectory_cloud->clear();
            if (vizData->insposes) {
                // Note: Assumes your StateHashMap stores a struct containing a gtsam::NavState named 'state'
                for (const auto& [id, ins_state_data] : *(vizData->insposes)) {
                    const auto& ins_state = ins_state_data.state; // Assumes KeyStateInfo has a 'state' member
                    pcl::PointXYZRGB p;
                    p.x = ins_state.pose().translation().x();
                    p.y = ins_state.pose().translation().y();
                    p.z = ins_state.pose().translation().z();
                    p.r = 30; p.g = 144; p.b = 255;
                    ins_trajectory_cloud->push_back(p);
                }
            }
            if (!viewer->updatePointCloud(ins_trajectory_cloud, "ins_trajectory_cloud")) {
                viewer->addPointCloud(ins_trajectory_cloud, "ins_trajectory_cloud");
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "ins_trajectory_cloud");
            }

            // --- 4. UPDATE CAMERA ---
            // if (!valid_ids.empty()) {
            //     uint64_t max_id = valid_ids.back();
            //     target_focal_point = vizData->poses->at<gtsam::Pose3>(gtsam::Symbol('x', max_id)).translation();
            // }
            // current_focal_point += (target_focal_point - current_focal_point) * kSmoothingFactor;
            // Eigen::Vector3d target_cam_pos = current_focal_point + kCameraOffset;
            // current_cam_pos += (target_cam_pos - current_cam_pos) * kSmoothingFactor;
            // viewer->setCameraPosition(
            //     current_cam_pos.x(), current_cam_pos.y(), current_cam_pos.z(),
            //     current_focal_point.x(), current_focal_point.y(), current_focal_point.z(),
            //     kUpVector.x(), kUpVector.y(), kUpVector.z()
            // );
            
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