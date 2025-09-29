#include <pipeline.hpp>

static std::atomic<bool> running = true;

void signal_handler(int) {
    running = false;
}

int main() {

    struct sigaction sa = {};
    sa.sa_handler = signal_handler;
    sigaction(SIGINT, &sa, nullptr);

    // std::string imuparam = "../config/imu_config_calib.json";
    std::string imuparam = "../config/imu_config_berlin.json";
             // initialize
    CompCallback compCallback(imuparam);
    FrameQueue<CompFrame> compData;
    
    auto compLastTs = std::make_shared<double>(0);

    boost::asio::io_context comp_iocontext;
    
    // UdpSocketConfig compUdpConfig;
    // compUdpConfig.host = "10.0.201.101";
    // compUdpConfig.multicastGroup = std::nullopt;
    // compUdpConfig.localInterfaceIp = "10.0.201.101";
    // compUdpConfig.port = 42001;
    // compUdpConfig.bufferSize = 105;
    // compUdpConfig.receiveTimeout = std::chrono::milliseconds(10000); 
    // compUdpConfig.reuseAddress = true; 
    // compUdpConfig.enableBroadcast = false; 
    // compUdpConfig.ttl =  std::nullopt; 

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
    auto comp_callback = [&compCallback, &compData, compLastTs](const DataBuffer& packet) {
        if (!running) return;
        auto frame = std::make_unique<CompFrame>();
        compCallback.Decode(packet, *frame); // Assuming Decode is a static method or external function
        if (frame->timestamp_20 > 0 && frame->timestamp_20 != *compLastTs) {
            std::cout << "Data frame push " << frame->timestamp_20 << " .\n";

            *compLastTs = frame->timestamp_20;
            compData.push(std::move(frame)); // Push the copy into compQueue
        }};
    //####################################################################################################
    auto comp_errcallback = [](const boost::system::error_code& ec) {
        if (running) {
            std::cerr << "Compass IO error: " << ec.message() << " (code: " << ec.value() << ")\n";
        }};
    //####################################################################################################
    auto comp_socket = std::shared_ptr<UdpSocket>(UdpSocket::create(comp_iocontext, compUdpConfig, comp_callback, comp_errcallback));
    //####################################################################################################
    auto comp_iothread = std::thread([&comp_iocontext, &compData]() {
        try {
            while (running) {
                comp_iocontext.run_one();
            }
        } catch (const std::exception& e) {
            std::cerr << "Compass IO context error: " << e.what() << "\n";
        }
        compData.stop();
    });
    //####################################################################################################
    auto calib_thread = std::thread([&compData]() {
        CompasHashMap compasArchive;
        uint64_t id = 209820;
        try {
            while (running) {
                auto data_frame = compData.pop();
                if (!data_frame) {
                    if (!running) std::cout << "Data frame stopped, exiting calibration thread.\n";
                    break;
                }
                id++;
                double ts = data_frame->timestamp_20;
                
                // Corrected: Use std::move to transfer ownership of the unique_ptr
                compasArchive[id] = {std::move(data_frame), ts, id};
            }
        } catch (const std::exception& e) {
            std::cerr << "Calibration thread error: " << e.what() << "\n";
        }

        if (!compasArchive.empty()) {
            std::cout << "\n[Calibration Thread] Loop finished. Writing " 
                    << compasArchive.size() 
                    << " compass frames to file...\n";
            
            // Call the function to write the compass data to a CSV file
            writeCompasToFile(compasArchive, "../compass.csv");
            
            std::cout << "[Calibration Thread] Compass data successfully written to compass.csv\n";
        }
        std::cout << "Calibration thread exiting\n";
    });

    // Cleanup
    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    comp_socket->stop();
    compData.stop();

    if (comp_iothread.joinable()) comp_iothread.join();
    if (calib_thread.joinable()) calib_thread.join();
    
    std::cout << "All threads have been joined. Shutdown complete." << std::endl;
}