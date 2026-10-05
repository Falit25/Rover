#include "lunar_slam/mobility_risk_model.hpp"
#include <rclcpp/rclcpp.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>

namespace lunar_slam {

// DecisionTree implementation
void MobilityRiskModel::DecisionTree::buildTree(Node* node, const std::vector<int>& indices, 
                                                const std::vector<std::vector<double>>& X, 
                                                const std::vector<std::array<double, 4>>& y, int depth) {
    if (depth >= max_depth || indices.size() < min_samples_split) {
        node->is_leaf = true;
        for (int i = 0; i < 4; ++i) {
            double sum = 0.0;
            for (int idx : indices) sum += y[idx][i];
            node->value[i] = sum / indices.size();
        }
        return;
    }
    
    auto [best_feature, best_threshold] = findBestSplit(indices, X, y);
    
    if (best_feature == -1) {
        node->is_leaf = true;
        for (int i = 0; i < 4; ++i) {
            double sum = 0.0;
            for (int idx : indices) sum += y[idx][i];
            node->value[i] = sum / indices.size();
        }
        return;
    }
    
    node->feature_idx = best_feature;
    node->threshold = best_threshold;
    node->is_leaf = false;
    
    std::vector<int> left_indices, right_indices;
    for (int idx : indices) {
        if (X[idx][best_feature] <= best_threshold) left_indices.push_back(idx);
        else right_indices.push_back(idx);
    }
    
    if (left_indices.empty() || right_indices.empty()) {
        node->is_leaf = true;
        for (int i = 0; i < 4; ++i) {
            double sum = 0.0;
            for (int idx : indices) sum += y[idx][i];
            node->value[i] = sum / indices.size();
        }
        return;
    }
    
    node->left = std::make_unique<Node>();
    node->right = std::make_unique<Node>();
    buildTree(node->left.get(), left_indices, X, y, depth + 1);
    buildTree(node->right.get(), right_indices, X, y, depth + 1);
}

std::pair<int, double> MobilityRiskModel::DecisionTree::findBestSplit(const std::vector<int>& indices,
                                             const std::vector<std::vector<double>>& X,
                                             const std::vector<std::array<double, 4>>& y) {
    int n_features = X[0].size();
    double best_gain = -1.0;
    int best_feature = -1;
    double best_threshold = 0.0;
    
    // Current impurity (average across 4 outputs)
    double current_impurity = 0.0;
    for (int i = 0; i < 4; ++i) {
        current_impurity += computeImpurity(indices, y, i);
    }
    current_impurity /= 4.0;
    
    std::uniform_int_distribution<int> feature_dist(0, n_features - 1);
    int n_features_to_try = std::max(1, n_features / 3);
    
    for (int f = 0; f < n_features_to_try; ++f) {
        int feature_idx = feature_dist(rng);
        
        // Get unique values for this feature
        std::vector<double> values;
        values.reserve(indices.size());
        for (int idx : indices) values.push_back(X[idx][feature_idx]);
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        
        if (values.size() < 2) continue;
        
        // Try a few threshold values
        for (size_t t = 1; t < values.size(); t += std::max(1, static_cast<int>(values.size()) / 10)) {
            double threshold = (values[t-1] + values[t]) / 2.0;
            
            std::vector<int> left_idx, right_idx;
            for (int idx : indices) {
                if (X[idx][feature_idx] <= threshold) left_idx.push_back(idx);
                else right_idx.push_back(idx);
            }
            
            if (left_idx.empty() || right_idx.empty()) continue;
            
            // Weighted impurity
            double weighted_impurity = 0.0;
            for (int i = 0; i < 4; ++i) {
                double left_imp = computeImpurity(left_idx, y, i);
                double right_imp = computeImpurity(right_idx, y, i);
                weighted_impurity += (left_idx.size() * left_imp + right_idx.size() * right_imp) / indices.size();
            }
            weighted_impurity /= 4.0;
            
            double gain = current_impurity - weighted_impurity;
            if (gain > best_gain) {
                best_gain = gain;
                best_feature = feature_idx;
                best_threshold = threshold;
            }
        }
    }
    
    return {best_feature, best_threshold};
}

double MobilityRiskModel::DecisionTree::computeImpurity(
    const std::vector<int>& indices,
    const std::vector<std::array<double, 4>>& y, int output_idx) const {
    
    if (indices.empty()) return 0.0;
    
    std::vector<double> values;
    values.reserve(indices.size());
    for (int idx : indices) values.push_back(y[idx][output_idx]);
    
    return computeVariance(values);
}

double MobilityRiskModel::DecisionTree::computeVariance(const std::vector<double>& values) const {
    if (values.size() < 2) return 0.0;
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double var = 0.0;
    for (double v : values) var += (v - mean) * (v - mean);
    return var / values.size();
}

void MobilityRiskModel::DecisionTree::fit(const std::vector<std::vector<double>>& X,
                                           const std::vector<std::array<double, 4>>& y) {
    std::vector<int> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    root = std::make_unique<Node>();
    buildTree(root.get(), indices, X, y, 0);
}

std::array<double, 4> MobilityRiskModel::DecisionTree::predict(const std::vector<double>& x) const {
    std::array<double, 4> result{};
    for (int i = 0; i < 4; ++i) {
        result[i] = predictSingle(x, i);
    }
    return result;
}

double MobilityRiskModel::DecisionTree::predictSingle(const std::vector<double>& x, int output_idx) const {
    const Node* node = root.get();
    while (node && !node->is_leaf) {
        if (x[node->feature_idx] <= node->threshold) {
            node = node->left.get();
        } else {
            node = node->right.get();
        }
    }
    return node ? node->value[output_idx] : 0.0;
}

// DecisionTree serialization methods
nlohmann::json MobilityRiskModel::DecisionTree::nodeToJson(const Node* node) const {
    if (!node) {
        return nullptr;
    }
    
    nlohmann::json j;
    j["feature_idx"] = node->feature_idx;
    j["threshold"] = node->threshold;
    j["value"] = {node->value[0], node->value[1], node->value[2], node->value[3]};
    j["is_leaf"] = node->is_leaf;
    j["n_samples"] = node->n_samples;
    j["impurity"] = node->impurity;
    
    if (node->left) j["left"] = nodeToJson(node->left.get());
    else j["left"] = nullptr;
    
    if (node->right) j["right"] = nodeToJson(node->right.get());
    else j["right"] = nullptr;
    
    return j;
}

std::unique_ptr<MobilityRiskModel::DecisionTree::Node> 
MobilityRiskModel::DecisionTree::nodeFromJson(const nlohmann::json& j) {
    if (j.is_null()) return nullptr;
    
    auto node = std::make_unique<Node>();
    node->feature_idx = j.value("feature_idx", -1);
    node->threshold = j.value("threshold", 0.0);
    node->value[0] = j["value"][0];
    node->value[1] = j["value"][1];
    node->value[2] = j["value"][2];
    node->value[3] = j["value"][3];
    node->is_leaf = j.value("is_leaf", false);
    node->n_samples = j.value("n_samples", 0);
    node->impurity = j.value("impurity", 0.0);
    
    if (j.contains("left") && !j["left"].is_null()) {
        node->left = nodeFromJson(j["left"]);
    }
    if (j.contains("right") && !j["right"].is_null()) {
        node->right = nodeFromJson(j["right"]);
    }
    
    return node;
}

nlohmann::json MobilityRiskModel::treeToJson(const DecisionTree& tree) const {
    nlohmann::json j;
    j["max_depth"] = tree.max_depth;
    j["min_samples_split"] = tree.min_samples_split;
    if (tree.root) j["root"] = nodeToJson(tree.root.get());
    return j;
}

std::unique_ptr<MobilityRiskModel::DecisionTree> 
MobilityRiskModel::treeFromJson(const nlohmann::json& j) {
    auto tree = std::make_unique<DecisionTree>(j.value("max_depth", 10), std::random_device{}());
    tree->min_samples_split = j.value("min_samples_split", 5);
    if (j.contains("root") && !j["root"].is_null()) {
        tree->root = nodeFromJson(j["root"]);
    }
    return tree;
}

// MobilityRiskModel implementation
MobilityRiskModel::MobilityRiskModel(const Config& config) : config_(config) {
    trees_.reserve(config_.n_estimators);
    for (int i = 0; i < config_.n_estimators; ++i) {
        trees_.emplace_back(std::make_unique<DecisionTree>(config_.max_depth, 
                                                           std::random_device{}()));
    }
    feature_importance_.resize(TerrainFeatures::DIM, 0.0);
}

MobilityRiskPrediction MobilityRiskModel::predict(const TerrainFeatures& features) const {
    MobilityRiskPrediction pred;
    
    if (!trained_ || trees_.empty()) {
        // Return prior estimates
        pred.slip_probability = 0.1;
        pred.sinkage_risk = 0.05;
        pred.traction_loss = 0.1;
        pred.immobilization_probability = 0.02;
        pred.confidence = 0.1;
        pred.prediction_uncertainty = 1.0;
        return pred;
    }
    
    std::vector<double> x = features.toVector();
    std::array<double, 4> sum = {0, 0, 0, 0};
    std::array<double, 4> sum_sq = {0, 0, 0, 0};
    
    for (const auto& tree : trees_) {
        auto pred_arr = tree->predict(x);
        for (int i = 0; i < 4; ++i) {
            sum[i] += pred_arr[i];
            sum_sq[i] += pred_arr[i] * pred_arr[i];
        }
    }
    
    int n = trees_.size();
    pred.slip_probability = std::clamp(sum[0] / n, 0.0, 1.0);
    pred.sinkage_risk = std::clamp(sum[1] / n, 0.0, 1.0);
    pred.traction_loss = std::clamp(sum[2] / n, 0.0, 1.0);
    pred.immobilization_probability = std::clamp(sum[3] / n, 0.0, 1.0);
    
    // Uncertainty from variance across trees
    double var_slip = (sum_sq[0] / n) - (sum[0] / n) * (sum[0] / n);
    double var_sink = (sum_sq[1] / n) - (sum[1] / n) * (sum[1] / n);
    double var_trac = (sum_sq[2] / n) - (sum[2] / n) * (sum[2] / n);
    double var_immob = (sum_sq[3] / n) - (sum[3] / n) * (sum[3] / n);
    
    pred.prediction_uncertainty = (var_slip + var_sink + var_trac + var_immob) / 4.0;
    pred.confidence = 1.0 / (1.0 + pred.prediction_uncertainty * 10.0);
    
    return pred;
}

std::vector<MobilityRiskPrediction> MobilityRiskModel::predictBatch(const std::vector<TerrainFeatures>& batch) const {
    std::vector<MobilityRiskPrediction> results;
    results.reserve(batch.size());
    for (const auto& f : batch) {
        results.push_back(predict(f));
    }
    return results;
}

void MobilityRiskModel::update(const TerrainFeatures& features, const MobilityRiskPrediction& ground_truth) {
    if (!config_.online_learning) return;
    
    X_buffer_.push_back(features.toVector());
    y_buffer_.push_back({ground_truth.slip_probability, ground_truth.sinkage_risk,
                         ground_truth.traction_loss, ground_truth.immobilization_probability});
    
    // Keep buffer bounded
    if (X_buffer_.size() > BUFFER_SIZE) {
        X_buffer_.erase(X_buffer_.begin());
        y_buffer_.erase(y_buffer_.begin());
    }
    
    updates_since_retrain_++;
    if (updates_since_retrain_ >= RETRAIN_INTERVAL && X_buffer_.size() >= 50) {
        // Retrain on buffered data
        std::vector<std::vector<double>> X_train = X_buffer_;
        std::vector<std::array<double, 4>> y_train = y_buffer_;
        
        // Retrain a subset of trees
        int n_retrain = std::min(config_.n_estimators / 10, (int)trees_.size());
        std::shuffle(trees_.begin(), trees_.end(), std::mt19937{std::random_device{}()});
        
        for (int i = 0; i < n_retrain; ++i) {
            trees_[i] = std::make_unique<DecisionTree>(config_.max_depth, std::random_device{}());
            trees_[i]->fit(X_train, y_train);
        }
        
        trained_ = true;
        updates_since_retrain_ = 0;
    }
}

std::vector<double> MobilityRiskModel::getFeatureImportance() const {
    return feature_importance_;
}

// DecisionTree serialization methods
nlohmann::json MobilityRiskModel::DecisionTree::nodeToJson(const Node* node) const {
    if (!node) {
        return nullptr;
    }
    
    nlohmann::json j;
    j["feature_idx"] = node->feature_idx;
    j["threshold"] = node->threshold;
    j["value"] = {node->value[0], node->value[1], node->value[2], node->value[3]};
    j["is_leaf"] = node->is_leaf;
    j["n_samples"] = node->n_samples;
    j["impurity"] = node->impurity;
    
    if (node->left) j["left"] = nodeToJson(node->left.get());
    else j["left"] = nullptr;
    
    if (node->right) j["right"] = nodeToJson(node->right.get());
    else j["right"] = nullptr;
    
    return j;
}

std::unique_ptr<MobilityRiskModel::DecisionTree::Node> 
MobilityRiskModel::DecisionTree::nodeFromJson(const nlohmann::json& j) {
    if (j.is_null()) return nullptr;
    
    auto node = std::make_unique<Node>();
    node->feature_idx = j.value("feature_idx", -1);
    node->threshold = j.value("threshold", 0.0);
    node->value[0] = j["value"][0];
    node->value[1] = j["value"][1];
    node->value[2] = j["value"][2];
    node->value[3] = j["value"][3];
    node->is_leaf = j.value("is_leaf", false);
    node->n_samples = j.value("n_samples", 0);
    node->impurity = j.value("impurity", 0.0);
    
    if (j.contains("left") && !j["left"].is_null()) {
        node->left = nodeFromJson(j["left"]);
    }
    if (j.contains("right") && !j["right"].is_null()) {
        node->right = nodeFromJson(j["right"]);
    }
    
    return node;
}

nlohmann::json MobilityRiskModel::treeToJson(const DecisionTree& tree) const {
    nlohmann::json j;
    j["max_depth"] = tree.max_depth;
    j["min_samples_split"] = tree.min_samples_split;
    if (tree.root) j["root"] = nodeToJson(tree.root.get());
    return j;
}

std::unique_ptr<MobilityRiskModel::DecisionTree> 
MobilityRiskModel::treeFromJson(const nlohmann::json& j) {
    auto tree = std::make_unique<DecisionTree>(j.value("max_depth", 10), std::random_device{}());
    tree->min_samples_split = j.value("min_samples_split", 5);
    if (j.contains("root") && !j["root"].is_null()) {
        tree->root = nodeFromJson(j["root"]);
    }
    return tree;
}

// Save/Load with full tree serialization
bool MobilityRiskModel::save(const std::string& path) const {
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
        
        const auto& cfg = j["config"];
        config_.n_estimators = cfg.value("n_estimators", config_.n_estimators);
        config_.max_depth = cfg.value("max_depth", config_.max_depth);
        config_.learning_rate = cfg.value("learning_rate", config_.learning_rate);
        config_.slip_threshold = cfg.value("slip_threshold", config_.slip_threshold);
        config_.sinkage_threshold = cfg.value("sinkage_threshold", config_.sinkage_threshold);
        
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

}