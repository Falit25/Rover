#pragma once

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <memory>
#include <random>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

namespace lunar_slam {

struct TerrainFeatures {
    // Geometric features
    double elevation = 0.0;
    double slope_deg = 0.0;
    double curvature = 0.0;
    double roughness = 0.0;
    double step_height = 0.0;
    
    // Thermal features
    double temperature_k = 250.0;
    double thermal_contrast = 0.0;
    double thermal_gradient = 0.0;
    
    // Rover state features
    double velocity = 0.0;
    double angular_velocity = 0.0;
    double wheel_torque = 0.0;
    
    // Convert to feature vector
    std::vector<double> toVector() const {
        return {elevation, slope_deg, curvature, roughness, step_height,
                temperature_k, thermal_contrast, thermal_gradient,
                velocity, angular_velocity, wheel_torque};
    }
    
    static constexpr int DIM = 11;
    static std::vector<std::string> featureNames() {
        return {"elevation", "slope_deg", "curvature", "roughness", "step_height",
                "temperature_k", "thermal_contrast", "thermal_gradient",
                "velocity", "angular_velocity", "wheel_torque"};
    }
};

struct MobilityRiskPrediction {
    double slip_probability = 0.0;           // P(wheel slip > threshold)
    double sinkage_risk = 0.0;               // Expected sinkage [0, 1]
    double traction_loss = 0.0;              // Traction coefficient reduction [0, 1]
    double immobilization_probability = 0.0; // P(stuck)
    double confidence = 0.0;                 // Model confidence [0, 1]
    double prediction_uncertainty = 0.0;     // Epistemic uncertainty
    
    // Combined risk score for planning
    double combined_risk() const {
        return 0.35 * slip_probability + 0.25 * sinkage_risk + 
               0.20 * traction_loss + 0.20 * immobilization_probability;
    }
};

class MobilityRiskModel {
public:
    struct Config {
        int n_estimators = 100;
        int max_depth = 10;
        double learning_rate = 0.1;
        double slip_threshold = 0.15;  // 15% slip ratio
        double sinkage_threshold = 0.05; // 5cm
        std::string model_path = "";
        bool use_thermal = true;
        bool online_learning = false;
        
        Config() = default;
    };
    
    explicit MobilityRiskModel(const Config& config = Config());
    
    // Predict mobility risk from terrain features
    MobilityRiskPrediction predict(const TerrainFeatures& features) const;
    
    // Online update with new observation
    void update(const TerrainFeatures& features, const MobilityRiskPrediction& ground_truth);
    
    // Save/load model
    bool save(const std::string& path) const;
    bool load(const std::string& path);
    
    // Get feature importance
    std::vector<double> getFeatureImportance() const;
    
    // Batch prediction for costmap
    std::vector<MobilityRiskPrediction> predictBatch(const std::vector<TerrainFeatures>& batch) const;
    
    bool isTrained() const { return trained_; }
    
private:
    Config config_;
    bool trained_ = false;
    
    // Simple Random Forest implementation (header-only, no external ML deps)
    struct DecisionTree {
        struct Node {
            int feature_idx = -1;
            double threshold = 0.0;
            double value[4] = {0.0, 0.0, 0.0, 0.0}; // slip, sinkage, traction, immobilization
            std::unique_ptr<Node> left;
            std::unique_ptr<Node> right;
            bool is_leaf = false;
            int n_samples = 0;
            double impurity = 0.0;
        };
        
        std::unique_ptr<Node> root;
        int max_depth;
        int min_samples_split = 5;
        std::mt19937 rng;
        
        DecisionTree(int max_depth, unsigned seed) : max_depth(max_depth), rng(seed) {}
        
        void fit(const std::vector<std::vector<double>>& X, const std::vector<std::array<double, 4>>& y);
        std::array<double, 4> predict(const std::vector<double>& x) const;
        double predictSingle(const std::vector<double>& x, int output_idx) const;
        
    public:  // Make serialization methods public
        nlohmann::json nodeToJson(const Node* node) const;
        std::unique_ptr<Node> nodeFromJson(const nlohmann::json& j);
        
    private:
        void buildTree(Node* node, const std::vector<int>& indices, 
                       const std::vector<std::vector<double>>& X, 
                       const std::vector<std::array<double, 4>>& y, int depth);
        std::pair<int, double> findBestSplit(const std::vector<int>& indices,
                                             const std::vector<std::vector<double>>& X,
                                             const std::vector<std::array<double, 4>>& y);
        double computeImpurity(const std::vector<int>& indices, 
                               const std::vector<std::array<double, 4>>& y, int output_idx) const;
        double computeVariance(const std::vector<double>& values) const;
    };
    
    std::vector<std::unique_ptr<DecisionTree>> trees_;
    std::vector<double> feature_importance_;
    
    // Training data buffer for online learning
    std::vector<std::vector<double>> X_buffer_;
    std::vector<std::array<double, 4>> y_buffer_;
    static constexpr int BUFFER_SIZE = 1000;
    static constexpr int RETRAIN_INTERVAL = 100;
    int updates_since_retrain_ = 0;
    
    // Serialization helpers
    nlohmann::json treeToJson(const DecisionTree& tree) const;
    std::unique_ptr<DecisionTree> treeFromJson(const nlohmann::json& j);
};

} 