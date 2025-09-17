#include <registercallback.hpp>

using json = nlohmann::json;

RegisterCallback::RegisterCallback(const std::string& json_param_path) {

    std::ifstream file_param(json_param_path);
    if (!file_param.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + json_param_path);
    }
    json json_param_data;
    try {
        file_param >> json_param_data;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("JSON parse error in " + json_param_path + ": " + e.what());
    }
    ParseParamdata(json_param_data);

}

RegisterCallback::RegisterCallback(const json& json_param) {
    ParseParamdata(json_param);
}

void RegisterCallback::ParseParamdata(const nlohmann::json& json_data) {
    if (!json_data.is_object()) {
        throw std::runtime_error("JSON data must be an object");
    }
    parameter_ = json_data;
    try {
        if (!json_data.contains("register_parameter") || !json_data["register_parameter"].is_object()) {
            throw std::runtime_error("Missing or invalid 'register_parameter' object");
        }
        const auto& register_param = json_data["register_parameter"];
        // Parse parameters if they exist in the JSON object
        if (register_param.contains("num_threads")) {
            num_threads_ = register_param["num_threads"].get<int>();
        }
        if (register_param.contains("mapvoxelsize")) {
            mapvoxelsize_ = register_param["mapvoxelsize"].get<float>();
        }
        if (register_param.contains("registration_method")) {
            registration_method_ = register_param["registration_method"].get<std::string>();
        }
        if (register_param.contains("ndt_resolution")) {
            ndt_resolution_ = register_param["ndt_resolution"].get<float>();
        }
        if (register_param.contains("ndt_transform_epsilon")) {
            ndt_transform_epsilon_ = register_param["ndt_transform_epsilon"].get<float>();
        }
        if (register_param.contains("ndt_neighborhood_search_method")) {
            ndt_neighborhood_search_method_ = register_param["ndt_neighborhood_search_method"].get<std::string>();
        }
        if (register_param.contains("gicp_corr_dist_threshold")) {
            gicp_corr_dist_threshold_ = register_param["gicp_corr_dist_threshold"].get<float>();
        }
        if (register_param.contains("gicp_transform_epsilon")) {
            gicp_transform_epsilon_ = register_param["gicp_transform_epsilon"].get<float>();
        }
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("JSON parsing error in register_parameter: " + std::string(e.what()));
    }
}

