#pragma once

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <vector>
#include <queue>
#include <unordered_map>
#include <limits>
#include <memory>

namespace lunar_slam {

struct RiskAwareConfig {
    double risk_weight = 2.0;
    double max_risk_per_step = 0.3;
    double risk_accumulation_threshold = 0.95;  // Renamed from cvar_alpha - risk budget for path
    int max_iterations = 10000;
    double heuristic_weight = 1.0;
    bool allow_diagonal = true;
    double diagonal_cost = 1.414;
    
    // Energy-aware planning
    double energy_weight = 1.5;
    double slope_energy_factor = 2.0;      // Energy multiplier per degree slope
    double slip_energy_factor = 3.0;       // Energy multiplier for slip risk
    double base_energy_per_meter = 1.0;    // Base energy cost per meter
    double max_energy_per_step = 50.0;     // Max energy budget per step
    double battery_capacity = 1000.0;      // Total energy budget (J)
    bool enable_energy_planning = true;
};

struct GridCell {
    int x, y;
    double g_cost = std::numeric_limits<double>::infinity();
    double h_cost = 0.0;
    double risk = 0.0;
    double risk_accum = 0.0;
    double energy = 0.0;
    double energy_accum = 0.0;
    int parent_x = -1, parent_y = -1;
    bool closed = false;

    double fCost() const { return g_cost + h_cost; }
    bool operator>(const GridCell& other) const { return fCost() > other.fCost(); }
};

class RiskAwarePlanner {
public:
    explicit RiskAwarePlanner(const RiskAwareConfig& config = RiskAwareConfig());

    nav_msgs::msg::Path planPath(const nav_msgs::msg::OccupancyGrid& costmap,
                                 const geometry_msgs::msg::Pose& start,
                                 const geometry_msgs::msg::Pose& goal,
                                 const std::vector<std::vector<double>>* risk_map = nullptr,
                                 const std::vector<std::vector<double>>* slope_map = nullptr,
                                 const std::vector<std::vector<double>>* slip_map = nullptr);

    void setConfig(const RiskAwareConfig& config) { config_ = config; }
    RiskAwareConfig getConfig() const { return config_; }

private:
    RiskAwareConfig config_;

    int worldToGridX(double wx, const nav_msgs::msg::MapMetaData& info) const;
    int worldToGridY(double wy, const nav_msgs::msg::MapMetaData& info) const;
    double gridToWorldX(int gx, const nav_msgs::msg::MapMetaData& info) const;
    double gridToWorldY(int gy, const nav_msgs::msg::MapMetaData& info) const;

    double heuristic(int x1, int y1, int x2, int y2) const;
    double computeCellRisk(int x, int y, const nav_msgs::msg::OccupancyGrid& costmap,
                           const std::vector<std::vector<double>>* risk_map) const;
    double computeCellEnergy(int x, int y, const nav_msgs::msg::OccupancyGrid& costmap,
                             const std::vector<std::vector<double>>* slope_map,
                             const std::vector<std::vector<double>>* slip_map) const;
    bool isValid(int x, int y, const nav_msgs::msg::OccupancyGrid& costmap) const;
    std::vector<std::pair<int, int>> getNeighbors(int x, int y) const;
    nav_msgs::msg::Path reconstructPath(const std::unordered_map<long, GridCell>& cells,
                                        int goal_x, int goal_y,
                                        const nav_msgs::msg::MapMetaData& info,
                                        const geometry_msgs::msg::Pose& start) const;
    long hash(int x, int y) const { return (static_cast<long>(x) << 32) | static_cast<unsigned int>(y); }
};

}  // namespace lunar_slam