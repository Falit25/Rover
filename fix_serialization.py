import re

with open('C:/SIH/mobility_risk_model.cpp', 'r') as f:
    content = f.read()

# Find the first save method and replace from there to the end
start = content.find('bool MobilityRiskModel::save(const std::string& path) const {')
if start == -1:
    print('NOT FOUND')
else:
    new_end = '''bool MobilityRiskModel::save(const std::string& path) const {
    if (!trained_) return false;
    
    nlohmann::json j;
    j["config"] = {
        {"n_estimators", config_.n_estimators},
        {"max_depth", config_.max_depth},
        {"learning_rate", config_.learning_rate},
        {"slip_threshold", config_.slip_threshold},
        {"sinkage_threshold", config_.sinkage_threshold}
    };
    j["trained"] = trained_;
    j["n_features"] = TerrainFeatures::DIM;
    
    // Serialize all trees
    nlohmann::json trees_json = nlohmann::json::array();
    for (const auto& tree : trees_) {
        trees_json.push_back(treeToJson(*tree));
    }
    j["trees"] = trees_json;
    
    // Feature importance
    j["feature_importance"] = feature_importance_;
    
    std::ofstream ofs(path);
    ofs << j.dump(2);
    return true;
}

bool MobilityRiskModel::load(const std::string& path) {
    try {
        std::ifstream ifs(path);
        nlohmann::json j;
        ifs >> j;
        
        config_.n_estimators = j.value("n_estimators", config_.n_estimators);
        config_.max_depth = j.value("max_depth", config_.max_depth);
        config_.learning_rate = j.value("learning_rate", config_.learning_rate);
        config_.slip_threshold = j.value("slip_threshold", config_.slip_threshold);
        config_.sinkage_threshold = j.value("sinkage_threshold", config_.sinkage_threshold);
        
        trained_ = j.value("trained", false);
        
        // Load trees
        if (j.contains("trees")) {
            trees_.clear();
            trees_.reserve(config_.n_estimators);
            for (const auto& tree_json : j["trees"]) {
                auto tree = treeFromJson(tree_json);
                if (tree) trees_.push_back(std::move(tree));
            }
        }
        
        // Load feature importance
        if (j.contains("feature_importance")) {
            feature_importance_ = j["feature_importance"].get<std::vector<double>>();
        }
        
        return trained_;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("MobilityRiskModel"), "Failed to load model: %s", e.what());
        return false;
    }
}

}  // namespace lunar_slam'''

    new_content = content[:start] + new_end
    with open('C:/SIH/mobility_risk_model.cpp', 'w') as f:
        f.write(new_content)
    print('DONE')