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

        // Trust Gain parameters defined here ---

        Eigen::Vector<double, 9> insCovScalingVector{1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2}; // High uncertainty for denied state
        bool was_ins_denied = false; // Assume we start in a denied state
        double ins_current_trust_factor = 1.0;
        const double ins_recovery_rate = 0.0045; // Trust regained over 1/0.02 = 50 keyframes
        const Eigen::Vector<double, 9> ins_full_trust_scaling_vector = Eigen::Vector<double, 9>::Ones();

        Eigen::Vector<double, 3> gnssCovScalingVector{1e2, 1e2, 1e2}; // High uncertainty for denied state
        bool was_gnss_denied = false; // Assume we start in a denied state
        double gnss_current_trust_factor = 1.0;
        const double gnss_recovery_rate = 0.0045; // Trust regained over 1/0.02 = 50 keyframes
        const Eigen::Vector<double, 3> full_trust_gnss_scaling_vector = Eigen::Vector<double, 3>::Ones();

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

        const int targetWinSize = 5;
        std::deque<uint64_t> targetID;

        // =================================================================================
        // C. STATE & ARCHIVE VARIABLES
        // =================================================================================
        // StateHashMap insStateArchive;
        PointsHashMap pointsArchive;
        PoseHashMap insPoseArchive;
        Eigen::Vector3d ins_rlla = Eigen::Vector3d::Zero();
        bool is_first_keyframe = true;
        bool use_const_vel = false;

        gtsam::Pose3 predTb2m;
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
                // const bool& insValid = (ins.GNSSFixStatus_20 > MIN_INS_FIX_STATUS || ins.timestamp_20 > 0.0 || ins.latitude_20 != 0.0 || ins.longitude_20 != 0.0);         // this check because reading is scalar but it give 0 rad.
                // const bool& gnssValid = (ins.GNSSFixStatus_29 > MIN_GNSS_FIX_STATUS || ins.timestamp_29 > 0.0 || ins.latitude_29 != 0.0 || ins.longitude_29 != 0.0);        // this check because reading is scalar but it give 0 rad.                                                                                         // add better condition for this check
                // const bool& lidarValid = (data_frame->points.pointsBody.size() > 0);   
                
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
                    // if (insValid) {
                        ins_rlla = ins_lla;
                        const gtsam::Point3 ins_tb2m{registerCallback.lla2ned(ins_lla.x(), ins_lla.y(), ins_lla.z(), ins_rlla.x(), ins_rlla.y(), ins_rlla.z())};
                        current_ins_state = gtsam::NavState{ins_Cb2m, ins_tb2m, ins_vNED};
                        gtsam::Vector3 ins_gravity(0.0, 0.0, 9.81);
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

                        predTb2m = insInitialPose;
                        gtsam::Vector3 currvned = currentEstimates.at<gtsam::Vector3>(gtsam::Symbol('v', id));
                        prev_bias_optimized = currentEstimates.at<gtsam::imuBias::ConstantBias>(gtsam::Symbol('b', id));
                        prev_state_optimized = gtsam::NavState(insInitialPose, currvned);
                        imu_preintegrator = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(imu_params, prev_bias_optimized);

                        pointsArchive[id] = {pointsBody, timestamp};
                        // insStateArchive[id] = {current_ins_state, timestamp};#
                        insPoseArchive[id] = {current_ins_state.pose().matrix(), timestamp};
                        is_first_keyframe = false;

                        targetID.push_back(id);

                        //3.5 push vizual
                        if (!currentEstimates.empty()) {
                            auto vizData = std::make_unique<VisualizationData>();
                            vizData->poses = std::make_shared<gtsam::Values>(currentEstimates);
                            vizData->points = std::make_shared<PointsHashMap>(pointsArchive);
                            // vizData->insposes = std::make_shared<StateHashMap>(insStateArchive);
                            vizData->insposes = std::make_shared<PoseHashMap>(insPoseArchive);
                            vizQueue.push(std::move(vizData));
                        }
                    // }
                } else { // --- 3. Main Loop for Subsequent Keyframes ---

                    const gtsam::Point3 ins_tb2m{registerCallback.lla2ned(ins_lla.x(), ins_lla.y(), ins_lla.z(), ins_rlla.x(), ins_rlla.y(), ins_rlla.z())};
                    current_ins_state = gtsam::NavState{ins_Cb2m, ins_tb2m, ins_vNED};

                    // 3.1. Integrate IMU measurements between the last keyframe and this one.
                    imu_preintegrator->resetIntegrationAndSetBias(prev_bias_optimized);
                    if (data_frame->ins.size() < 2) {
                        std::cout << "Warning: Not enough IMU measurements to integrate between frames. Skipping integration." << std::endl;
                    } else {
                        // Initialize last_timestamp with the timestamp of the FIRST measurement
                        double last_timestamp = data_frame->ins.front().timestamp_20;

                        // Start the loop from the SECOND measurement (index 1)
                        for (size_t i = 1; i < data_frame->ins.size(); ++i) {
                            const auto& measurement = data_frame->ins[i];
                            double dt = measurement.timestamp_20 - last_timestamp;

                            // The dt > 0 check is still a good safety measure
                            if (dt > 0) {
                                gtsam::Vector3 accel(measurement.accelX_28, measurement.accelY_28, measurement.accelZ_28);
                                gtsam::Vector3 gyro(measurement.gyroX_28, measurement.gyroY_28, measurement.gyroZ_28);
                                imu_preintegrator->integrateMeasurement(accel, gyro, dt);
                            }
                            last_timestamp = measurement.timestamp_20;
                        }
                    }

                    // // 3.2. PREDICT the current state using the IMU preintegration.
                    // // This provides a high-quality initial guess for the new variables.
                    gtsam::NavState predicted_state = imu_preintegrator->predict(prev_state_optimized, prev_bias_optimized);
                    newEstimates.insert(gtsam::Symbol('x', id), predicted_state.pose());
                    newEstimates.insert(gtsam::Symbol('v', id), predicted_state.velocity());
                    newEstimates.insert(gtsam::Symbol('b', id), prev_bias_optimized); // Assume bias is constant for the initial guess
                    
                    // // 3.3. Add the IMU factor to the graph.
                    newFactors.add(gtsam::CombinedImuFactor(
                        gtsam::Symbol('x', targetID.back()), gtsam::Symbol('v', targetID.back()),
                        gtsam::Symbol('x', id), gtsam::Symbol('v', id),
                        gtsam::Symbol('b', targetID.back()), gtsam::Symbol('b', id),
                        *imu_preintegrator));
                    // if (insValid){
                        Eigen::Vector<double, 9> insStdDev = Eigen::Vector<double, 9>::Zero();
                        insStdDev << ins.sigmaLatitude_20, ins.sigmaLongitude_20, ins.sigmaAltitude_20, ins.sigmaRoll_26, ins.sigmaPitch_26, ins.sigmaYaw_26, ins.sigmaVelocityNorth_25, ins.sigmaVelocityEast_25, ins.sigmaVelocityDown_25;
                        double insChecker = insStdDev.head(3).norm();
                        bool is_ins_available_now = (insChecker < 0.25);
                        if (is_ins_available_now && was_ins_denied) {
                            std::cout << "Warning: INS return from denied position.start trust gain recovery.\n";
                            ins_current_trust_factor = 0.0; // Reset to begin recovery from zero trust
                        }
                        was_ins_denied = !is_ins_available_now;
                        Eigen::Vector<double, 9> current_ins_scaling_vector;
                        if (is_ins_available_now) {
                            // If available, increase trust factor and interpolate the scaling vector.
                            ins_current_trust_factor = std::min(1.0, ins_current_trust_factor + ins_recovery_rate);
                            current_ins_scaling_vector = insCovScalingVector + ins_current_trust_factor * (ins_full_trust_scaling_vector - insCovScalingVector);
                            std::cout << "Logging: INS Available. Current ins scalling factor.\n" << current_ins_scaling_vector.transpose() << std::endl;;
                        } else {
                            // If denied, reset trust and use the high uncertainty scaling.
                            std::cout << "Warning: INS Denied. Using low-trust covariance.\n";
                            // current_trust_factor = 0.0;
                            current_ins_scaling_vector = insCovScalingVector;
                        }
                        gtsam::Vector6 ins_scaled_sigmas;
                        ins_scaled_sigmas << insStdDev(3) * current_ins_scaling_vector(3), // roll
                                        insStdDev(4) * current_ins_scaling_vector(4), // pitch
                                        insStdDev(5) * current_ins_scaling_vector(5), // yaw
                                        insStdDev(0) * current_ins_scaling_vector(0), // x (from lat)
                                        insStdDev(1) * current_ins_scaling_vector(1), // y (from lon)
                                        insStdDev(2) * current_ins_scaling_vector(2);

                        // gtsam::Vector3 ins_vel_scaled_sigmas;
                        // ins_vel_scaled_sigmas << insStdDev(6) * current_ins_scaling_vector(6), // roll
                        //                 insStdDev(7) * current_ins_scaling_vector(7), // pitch
                        //                 insStdDev(8) * current_ins_scaling_vector(8);

                        gtsam::Pose3 insFactor = current_ins_state.pose();
                        gtsam::SharedNoiseModel insNoiseModel = gtsam::noiseModel::Diagonal::Sigmas(ins_scaled_sigmas);
                        newFactors.add(gtsam::PriorFactor<gtsam::Pose3>(Symbol('x', id), insFactor, insNoiseModel));

                        // gtsam::Vector3 insVelFactor = current_ins_state.velocity();
                        // gtsam::SharedNoiseModel insVelNoiseModel = gtsam::noiseModel::Diagonal::Sigmas(ins_vel_scaled_sigmas);
                        // newFactors.add(gtsam::PriorFactor<gtsam::Vector3>(Symbol('v', id),insVelFactor, insVelNoiseModel));
                    // }

                    if (!use_const_vel){
                        use_const_vel = true;
                    } else {
                        gtsam::Vector6 cv_scaled_sigmas;
                        cv_scaled_sigmas << 0.02, 0.02, 0.02, 0.2, 0.2, 0.2;
                        gtsam::SharedNoiseModel cvNoiseModel = gtsam::noiseModel::Diagonal::Sigmas(cv_scaled_sigmas);
                        newFactors.add(gtsam::PriorFactor<gtsam::Pose3>(Symbol('x', id), predTb2m, cvNoiseModel));
                    }
                    
                    // 3.4. (Conceptual) Add Lidar Odometry Factor
                    // if (lidarValid){
                        pcl::PointCloud<pcl::PointXYZI>::Ptr lidarFactorPointsTarget(new pcl::PointCloud<pcl::PointXYZI>());
                        for (const int& currID : targetID) {
                            pcl::PointCloud<pcl::PointXYZI>::Ptr currlidarFactorPointsTarget(new pcl::PointCloud<pcl::PointXYZI>());
                            const auto& currlidarFactorPointsArchive = pointsArchive.at(currID);
                            gtsam::Pose3 currlidarFactorTargetTb2m = currentEstimates.at<gtsam::Pose3>(Symbol('x', currID));
                            pcl::transformPointCloud(*currlidarFactorPointsArchive.points, *currlidarFactorPointsTarget, currlidarFactorTargetTb2m.matrix());
                            *lidarFactorPointsTarget += *currlidarFactorPointsTarget;
                        }
                        gtsam::Pose3 lidarFactorTargetTb2m = currentEstimates.at<gtsam::Pose3>(Symbol('x', targetID.back()));
                        pcl::PointCloud<pcl::PointXYZI>::Ptr lidarFactorPointsSource(new pcl::PointCloud<pcl::PointXYZI>());
                        ndt_omp->setInputTarget(lidarFactorPointsTarget);
                        ndt_omp->setInputSource(pointsBody);
                        ndt_omp->align(*lidarFactorPointsSource, predicted_state.pose().matrix().cast<float>());
                        gtsam::Pose3 lidarFactorSourceTb2m(ndt_omp->getFinalTransformation().cast<double>());
                        gtsam::Pose3 lidarTbs2bt = lidarFactorTargetTb2m.between(lidarFactorSourceTb2m);
                        
                        auto ndt_result = ndt_omp->getResult();
                        ndt_iter = ndt_result.iteration_num;
                        const auto& hessian = ndt_result.hessian;
                        Eigen::Matrix<double, 6, 6> regularized_hessian = hessian + (Eigen::Matrix<double, 6, 6>::Identity() * 1e-6);
                        Eigen::Matrix<double, 6, 6> lidar_cov = -regularized_hessian.inverse();
                        gtsam::SharedNoiseModel lidarNoiseModel = gtsam::noiseModel::Gaussian::Covariance(registerCallback.reorderCovarianceForGTSAM(lidar_cov));
                        newFactors.add(gtsam::BetweenFactor<gtsam::Pose3>(Symbol('x', targetID.back()), Symbol('x', id), lidarTbs2bt, lidarNoiseModel));
                    // }
                    // 3.4. (Conceptual) Add GPS Factor
                    // if(gnssValid){
                        // Eigen::Vector<double, 3> gnssStdDev = Eigen::Vector<double, 3>::Zero();
                        // gnssStdDev << ins.sigmaLatitude_29, ins.sigmaLongitude_29, ins.sigmaAltitude_29;
                        // double gnssChecker = gnssStdDev.norm();
                        // bool is_gnss_available_now = (gnssChecker < 1.0);
                        // if (is_gnss_available_now && was_gnss_denied) {
                        //     std::cout << "Warning: GPS return from denied position.start trust gain recovery.\n";
                        //     gnss_current_trust_factor = 0.0; // Reset to begin recovery from zero trust
                        // }
                        // was_gnss_denied = !is_gnss_available_now;
                        // Eigen::Vector<double, 3> current_gnss_scaling_vector;
                        // if (is_gnss_available_now) {
                        //     // If available, increase trust factor and interpolate the scaling vector.
                        //     gnss_current_trust_factor = std::min(1.0, gnss_current_trust_factor + gnss_recovery_rate);
                        //     current_gnss_scaling_vector = gnssCovScalingVector + gnss_current_trust_factor * (full_trust_gnss_scaling_vector - gnssCovScalingVector);
                        //     std::cout << "Logging: GNSS Available. Current ins scalling factor.\n" << current_gnss_scaling_vector.transpose() << std::endl;;
                        // } else {
                        //     // If denied, reset trust and use the high uncertainty scaling.
                        //     std::cout << "Warning: GNSS Denied. Using low-trust covariance.\n";
                        //     // current_trust_factor = 0.0;
                        //     current_gnss_scaling_vector = gnssCovScalingVector;
                        // }
                        // gtsam::Vector3 gnss_scaled_sigmas;
                        // gnss_scaled_sigmas << gnssStdDev(0) * current_gnss_scaling_vector(0), 
                        //              gnssStdDev(1) * current_gnss_scaling_vector(1), 
                        //              gnssStdDev(2) * current_gnss_scaling_vector(2);
                        
                        // const Eigen::Vector3d gnss_lla{ins.latitude_29, ins.longitude_29, ins.altitude_29};
                        // const gtsam::Point3 gnss_tb2m{registerCallback.lla2ned(gnss_lla.x(), gnss_lla.y(), gnss_lla.z(), ins_rlla.x(), ins_rlla.y(), ins_rlla.z())};
                        // gtsam::SharedNoiseModel gnssNoiseModel = gtsam::noiseModel::Diagonal::Sigmas(gnss_scaled_sigmas);
                        // newFactors.add(gtsam::GPSFactor(Symbol('x', id), gnss_tb2m, gnssNoiseModel));
                    // }
                    
                    isam2.update(newFactors, newEstimates);
                    currentEstimates = isam2.calculateEstimate();
                    gtsam::Pose3 currTb2m = currentEstimates.at<gtsam::Pose3>(Symbol('x', id));
                    gtsam::Pose3 prevTb2m = currentEstimates.at<gtsam::Pose3>(Symbol('x', targetID.back()));
                    gtsam::Vector3 currvned = currentEstimates.at<gtsam::Vector3>(Symbol('v', id));
                    Eigen::Matrix4d Tbc2bp = prevTb2m.matrix().inverse() * currTb2m.matrix();
                    // // gtsam::Pose3 Tbc2bp = prevTb2m.between(currTb2m);
                    predTb2m = gtsam::Pose3{currTb2m.matrix() * Tbc2bp};
                    prev_state_optimized = gtsam::NavState(currTb2m, currvned);
                    prev_bias_optimized = currentEstimates.at<gtsam::imuBias::ConstantBias>(Symbol('b', id));

                    // =========================================================================
                    // --- UPDATED: FULL POSE COMPARISON OUTPUT ---
                    // =========================================================================
                    std::cout << "\n--- POSE COMPARISON (Frame ID: " << id << ") ---\n";
                    const auto& raw_pose = current_ins_state.pose();
                    const gtsam::Vector3 raw_rpy_deg = raw_pose.rotation().rpy() * 180.0 / M_PI;
                    std::cout << "Raw INS Pose:     T = [" << raw_pose.translation().transpose() << "]\n"
                            << "                  RPY(deg) = [" << raw_rpy_deg.transpose() << "]\n";
                    
                    const auto& opt_pose = prev_state_optimized.pose();
                    const gtsam::Vector3 opt_rpy_deg = opt_pose.rotation().rpy() * 180.0 / M_PI;
                    std::cout << "GTSAM Opt. Pose:  T = [" << opt_pose.translation().transpose() << "]\n"
                            << "                  RPY(deg) = [" << opt_rpy_deg.transpose() << "]\n";
                    // =========================================================================

                    pointsArchive[id] = {pointsBody, timestamp};
                    // insStateArchive[id] = {current_ins_state, timestamp};
                    insPoseArchive[id] = {current_ins_state.pose().matrix(), timestamp}; 

                    if (targetID.size() >= targetWinSize) {
                        targetID.pop_front();
                    }
                    targetID.push_back(id);

                    //3.5 push vizual
                    if (!currentEstimates.empty()) {
                        auto vizData = std::make_unique<VisualizationData>();
                        vizData->poses = std::make_shared<gtsam::Values>(currentEstimates);
                        vizData->points = std::make_shared<PointsHashMap>(pointsArchive);
                        // vizData->insposes = std::make_shared<StateHashMap>(insStateArchive);
                        vizData->insposes = std::make_shared<PoseHashMap>(insPoseArchive);
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
    // auto viz_thread = std::thread([&vizQueue]() {
    //     auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("GTSAM Optimized Map");
    //     viewer->setBackgroundColor(0.1, 0.1, 0.1);
    //     viewer->addCoordinateSystem(10.0, "world_origin");
    //     viewer->initCameraParameters();

    //     // --- NEW: Camera Following & Smoothing Logic ---
    //     // This factor controls how quickly the camera catches up to the target.
    //     // Lower values (e.g., 0.05) are smoother but have more lag.
    //     // Higher values (e.g., 0.2) are more responsive but can be jumpy.
    //     const double kSmoothingFactor = 0.1;

    //     // This defines the camera's position relative to the focal point (view from above).
    //     const Eigen::Vector3d kCameraOffset(0.0, 0.0, -250.0);

    //     // The "up" vector for the camera. Your original code used (1,0,0), which is non-standard but preserved here.
    //     // A more common "up" vector would be (0, -1, 0) for Z-forward or (0, 0, 1) for Y-forward systems.
    //     const Eigen::Vector3d kUpVector(1.0, 0.0, 0.0);

    //     // State variables to hold the camera's current and target focus points.
    //     // Initialize them to the starting view.
    //     Eigen::Vector3d target_focal_point(0.0, 0.0, 0.0);
    //     Eigen::Vector3d current_focal_point = target_focal_point;
    //     Eigen::Vector3d current_cam_pos = target_focal_point + kCameraOffset;

    //     // --- MODIFIED: The initial setCameraPosition is now managed by the loop ---
    //     // viewer->setCameraPosition(0, 0, -50, 0, 0, 0, 1, 0, 0); // This is now handled dynamically

    //     const size_t kSlidingWindowSize = 50;
    //     std::deque<uint64_t> displayed_frame_ids;
    //     uint64_t last_processed_id = 0;

    //     pcl::VoxelGrid<pcl::PointXYZI> vg;
    //     vg.setLeafSize(0.5f, 0.5f, 0.5f);

    //     // Point cloud for the OPTIMIZED trajectory (colored RED)
    //     pcl::PointCloud<pcl::PointXYZRGB>::Ptr trajectory_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    //     // Point cloud for the RAW INS trajectory (colored GREEN)
    //     pcl::PointCloud<pcl::PointXYZRGB>::Ptr ins_trajectory_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    //     while (running && !viewer->wasStopped()) {
    //         auto vizData = vizQueue.pop();
    //         if (!vizData) {
    //             if (!running) std::cout << "Visualization queue stopped, exiting viz thread.\n";
    //             break;
    //         }

    //         gtsam::Pose3 latest_pose;
    //         uint64_t max_id = 0;
    //         if (!vizData->poses->empty()) { // Ensure poses are not empty before finding max
    //             for (const auto& key_value : *(vizData->poses)) {
    //                 uint64_t frame_id = gtsam::Symbol(key_value.key).index();
    //                 if (frame_id > max_id) {
    //                     max_id = frame_id;
    //                 }
    //             }
    //         }
            
    //         if (max_id > 0 && max_id != last_processed_id) {
    //             last_processed_id = max_id;
    //             latest_pose = vizData->poses->at<gtsam::Pose3>(Symbol('x', max_id));
                
    //             // --- NEW: Update the target focal point when a new pose is available ---
    //             target_focal_point = latest_pose.translation();

    //             auto it = vizData->points->find(max_id);
    //             if (it != vizData->points->end()) {
    //                 // ... (your existing point cloud processing logic is unchanged)
    //                 if (displayed_frame_ids.size() >= kSlidingWindowSize) {
    //                     uint64_t id_to_remove = displayed_frame_ids.front();
    //                     std::string cloud_id_to_remove = "map_cloud_" + std::to_string(id_to_remove);
    //                     if (viewer->contains(cloud_id_to_remove)) {
    //                         viewer->removePointCloud(cloud_id_to_remove);
    //                     }
    //                     displayed_frame_ids.pop_front();
    //                 }

    //                 pcl::PointCloud<pcl::PointXYZI>::Ptr raw_cloud = it->second.points;
    //                 pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    //                 pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    //                 pcl::PointCloud<pcl::PointXYZI>::Ptr spatial_cloud(new pcl::PointCloud<pcl::PointXYZI>());
                    
    //                 pcl::transformPointCloud(*raw_cloud, *transformed_cloud, latest_pose.matrix().cast<float>());
    //                 vg.setInputCloud(transformed_cloud);
    //                 vg.filter(*downsampled_cloud);
                    
    //                 pcl::PassThrough<pcl::PointXYZI> pass_spatial;
    //                 pass_spatial.setFilterFieldName("z");
    //                 pass_spatial.setFilterLimits(-300.0, 0.0);
    //                 pass_spatial.setInputCloud(downsampled_cloud);
    //                 pass_spatial.filter(*spatial_cloud);

    //                 std::string cloud_id_to_add = "map_cloud_" + std::to_string(max_id);
    //                 pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color_handler(spatial_cloud, "intensity");
    //                 viewer->addPointCloud(spatial_cloud, color_handler, cloud_id_to_add);
    //                 viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cloud_id_to_add);
                    
    //                 displayed_frame_ids.push_back(max_id);
    //             }
    //         }

    //         // --- 3. UPDATE TRAJECTORIES ---
    //         // Optimized GTSAM Trajectory (Deep Pink)
    //         trajectory_cloud->clear();
    //         for (const auto& key_value : *(vizData->poses)) {
    //             gtsam::Symbol symbol(key_value.key);
    //             if (symbol.chr() == 'x') { // Process only pose variables
    //                 gtsam::Pose3 pose = key_value.value.cast<gtsam::Pose3>();
    //                 pcl::PointXYZRGB p;
    //                 p.x = pose.translation().x(); p.y = pose.translation().y(); p.z = pose.translation().z();
    //                 p.r = 255; p.g = 20; p.b = 147;
    //                 trajectory_cloud->push_back(p);
    //             }
    //         }
    //         if (!viewer->updatePointCloud(trajectory_cloud, "trajectory_cloud")) {
    //             viewer->addPointCloud(trajectory_cloud, "trajectory_cloud");
    //             viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "trajectory_cloud");
    //         }

    //         // Raw INS Trajectory (Deep Sky Blue)
    //         ins_trajectory_cloud->clear();
    //         if (vizData->insposes) {
    //             // Note: Assumes your StateHashMap stores a struct containing a gtsam::NavState named 'state'
    //             for (const auto& [id, ins_state_data] : *(vizData->insposes)) {
    //                 const auto& ins_state = ins_state_data.state; // Assumes KeyStateInfo has a 'state' member
    //                 pcl::PointXYZRGB p;
    //                 p.x = ins_state.pose().translation().x();
    //                 p.y = ins_state.pose().translation().y();
    //                 p.z = ins_state.pose().translation().z();
    //                 p.r = 30; p.g = 144; p.b = 255;
    //                 ins_trajectory_cloud->push_back(p);
    //             }
    //         }
    //         if (!viewer->updatePointCloud(ins_trajectory_cloud, "ins_trajectory_cloud")) {
    //             viewer->addPointCloud(ins_trajectory_cloud, "ins_trajectory_cloud");
    //             viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "ins_trajectory_cloud");
    //         }

    //         // Interpolate the focal point
    //         current_focal_point = current_focal_point + (target_focal_point - current_focal_point) * kSmoothingFactor;
    //         // Calculate the desired camera position based on the smoothed focal point
    //         Eigen::Vector3d target_cam_pos = current_focal_point + kCameraOffset;
    //         // Interpolate the camera's actual position
    //         current_cam_pos = current_cam_pos + (target_cam_pos - current_cam_pos) * kSmoothingFactor;

    //         // Set the new camera position and view direction in the visualizer
    //         viewer->setCameraPosition(
    //             current_cam_pos.x(), current_cam_pos.y(), current_cam_pos.z(),
    //             current_focal_point.x(), current_focal_point.y(), current_focal_point.z(),
    //             kUpVector.x(), kUpVector.y(), kUpVector.z()
    //         );
            
    //         viewer->spinOnce(100);
    //     }
    //     std::cout << "Visualization thread exiting\n";
    // });
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

        const size_t kSlidingWindowSize = 20;
        std::deque<uint64_t> displayed_frame_ids;
        uint64_t last_processed_id = 0;

        pcl::VoxelGrid<pcl::PointXYZI> vg;
        vg.setLeafSize(0.5f, 0.5f, 0.5f);

        // Point cloud for the OPTIMIZED trajectory (colored RED)
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr trajectory_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        // Point cloud for the RAW INS trajectory (colored GREEN)
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ins_trajectory_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

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
                gtsam::Symbol symbol(key_value.key);

                // Add this check to process ONLY pose variables
                if (symbol.chr() == 'x') { 
                    gtsam::Pose3 pose = key_value.value.cast<gtsam::Pose3>();
                    pcl::PointXYZRGB trajectory_point;
                    trajectory_point.x = pose.translation().x();
                    trajectory_point.y = pose.translation().y();
                    trajectory_point.z = pose.translation().z();
                    trajectory_point.r = 255;
                    trajectory_point.g = 20;
                    trajectory_point.b = 147;
                    trajectory_cloud->push_back(trajectory_point);
                }
            }

            if (!viewer->updatePointCloud(trajectory_cloud, "trajectory_cloud")) {
                viewer->addPointCloud(trajectory_cloud, "trajectory_cloud");
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "trajectory_cloud");
            }

            // Update RAW INS trajectory (GREEN)
            ins_trajectory_cloud->clear();
            if (vizData->insposes) {
                for (const auto& key_value : *(vizData->insposes)) {
                    const auto& pose_matrix = key_value.second.pose;
                    pcl::PointXYZRGB ins_point;
                    ins_point.x = pose_matrix(0, 3);
                    ins_point.y = pose_matrix(1, 3);
                    ins_point.z = pose_matrix(2, 3);
                    ins_point.r = 30;
                    ins_point.g = 144;
                    ins_point.b = 255;
                    ins_trajectory_cloud->push_back(ins_point);
                }
            }

            if (!viewer->updatePointCloud(ins_trajectory_cloud, "ins_trajectory_cloud")) {
                viewer->addPointCloud(ins_trajectory_cloud, "ins_trajectory_cloud");
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "ins_trajectory_cloud");
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
    
    std::cout << "All threads have been joined. Shutdown complete." << std::endl;
}