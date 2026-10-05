#include "lunar_slam/risk_aware_planner.hpp"
#include <cmath>
#include <algorithm>

namespace lunar_slam {

RiskAwarePlanner::RiskAwarePlanner(const RiskAwareConfig& config) : config_(config) {}

nav_msgs::msg::Path RiskAwarePlanner::planPath(const nav_msgs::msg::OccupancyGrid& costmap,
                                               const geometry_msgs::msg::Pose& start,
                                               const geometry_msgs::msg::Pose& goal,
                                               const std::vector<std::vector<double>>* risk_map,
                                               const std::vector<std::vector<double>>* slope_map,
                                               const std::vector<std::vector<double>>* slip_map) {
    nav_msgs::msg::Path path_msg;
    path_msg.header = costmap.header;

    int width = costmap.info.width;
    int height = costmap.info.height;
    double res = costmap.info.resolution;
    double origin_x = costmap.info.origin.position.x;
    double origin_y = costmap.info.origin.position.y;

    int start_x = worldToGridX(start.position.x, costmap.info);
    int start_y = worldToGridY(start.position.y, costmap.info);
    int goal_x = worldToGridX(goal.position.x, costmap.info);
    int goal_y = worldToGridY(goal.position.y, costmap.info);

    if (!isValid(start_x, start_y, costmap) || !isValid(goal_x, goal_y, costmap)) {
        return path_msg;
    }

    std::priority_queue<GridCell, std::vector<GridCell>, std::greater<GridCell>> open_set;
    std::unordered_map<long, GridCell> all_cells;

    GridCell start_cell;
    start_cell.x = start_x;
    start_cell.y = start_y;
    start_cell.g_cost = 0.0;
    start_cell.h_cost = heuristic(start_x, start_y, goal_x, goal_y);
    start_cell.risk = 0.0;
    start_cell.risk_accum = 0.0;
    start_cell.energy = 0.0;
    start_cell.energy_accum = 0.0;

    open_set.push(start_cell);
    all_cells[hash(start_x, start_y)] = start_cell;

    int iterations = 0;
    while (!open_set.empty() && iterations < config_.max_iterations) {
        ++iterations;
        GridCell current = open_set.top();
        open_set.pop();

        long curr_hash = hash(current.x, current.y);
        auto it = all_cells.find(curr_hash);
        if (it == all_cells.end() || it->second.closed) continue;
        it->second.closed = true;

        if (current.x == goal_x && current.y == goal_y) {
            return reconstructPath(all_cells, goal_x, goal_y, costmap.info, start);
        }

        for (auto [nx, ny] : getNeighbors(current.x, current.y)) {
            if (!isValid(nx, ny, costmap)) continue;

            long n_hash = hash(nx, ny);
            auto n_it = all_cells.find(n_hash);
            if (n_it != all_cells.end() && n_it->second.closed) continue;

            double step_cost = (nx == current.x || ny == current.y) ? 1.0 : config_.diagonal_cost;
            step_cost *= res;

            double cell_risk = computeCellRisk(nx, ny, costmap, risk_map);
            
            // Reject individual cells with too high risk
            if (cell_risk > config_.max_risk_per_step) {
                continue;
            }
            
            double new_risk_accum = current.risk_accum + cell_risk;
            
            // Reject paths that exceed total risk budget
            if (new_risk_accum > config_.risk_accumulation_threshold) {
                continue;
            }

            // Energy cost computation
            double cell_energy = 0.0;
            if (config_.enable_energy_planning) {
                cell_energy = computeCellEnergy(nx, ny, costmap, slope_map, slip_map);
                double new_energy_accum = current.energy_accum + cell_energy;
                
                if (new_energy_accum > config_.max_energy_per_step * (current.g_cost + step_cost) / res) {
                    continue;
                }
            }

            // Combined cost: distance + risk_weight * risk + energy_weight * energy
            double new_g = current.g_cost + step_cost + 
                          config_.risk_weight * cell_risk +
                          config_.energy_weight * cell_energy;

            if (n_it == all_cells.end() || new_g < n_it->second.g_cost) {
                GridCell neighbor;
                neighbor.x = nx;
                neighbor.y = ny;
                neighbor.g_cost = new_g;
                neighbor.h_cost = config_.heuristic_weight * heuristic(nx, ny, goal_x, goal_y);
                neighbor.risk = cell_risk;
                neighbor.risk_accum = new_risk_accum;
                neighbor.energy = cell_energy;
                neighbor.energy_accum = current.energy_accum + cell_energy;
                neighbor.parent_x = current.x;
                neighbor.parent_y = current.y;

                all_cells[n_hash] = neighbor;
                open_set.push(neighbor);
            }
        }
    }

    return path_msg;
}

int RiskAwarePlanner::worldToGridX(double wx, const nav_msgs::msg::MapMetaData& info) const {
    return static_cast<int>(std::floor((wx - info.origin.position.x) / info.resolution));
}

int RiskAwarePlanner::worldToGridY(double wy, const nav_msgs::msg::MapMetaData& info) const {
    return static_cast<int>(std::floor((wy - info.origin.position.y) / info.resolution));
}

double RiskAwarePlanner::gridToWorldX(int gx, const nav_msgs::msg::MapMetaData& info) const {
    return info.origin.position.x + (gx + 0.5) * info.resolution;
}

double RiskAwarePlanner::gridToWorldY(int gy, const nav_msgs::msg::MapMetaData& info) const {
    return info.origin.position.y + (gy + 0.5) * info.resolution;
}

double RiskAwarePlanner::heuristic(int x1, int y1, int x2, int y2) const {
    double dx = std::abs(x1 - x2);
    double dy = std::abs(y1 - y2);
    if (config_.allow_diagonal) {
        return std::min(dx, dy) * config_.diagonal_cost + std::abs(dx - dy);
    }
    return dx + dy;
}

double RiskAwarePlanner::computeCellRisk(int x, int y, const nav_msgs::msg::OccupancyGrid& costmap,
                                         const std::vector<std::vector<double>>* risk_map) const {
    if (risk_map && y < static_cast<int>(risk_map->size()) && x < static_cast<int>((*risk_map)[0].size())) {
        return (*risk_map)[y][x];
    }

    int idx = y * costmap.info.width + x;
    if (idx >= 0 && idx < static_cast<int>(costmap.data.size())) {
        int8_t val = costmap.data[idx];
        if (val == -1) return 0.9;  // Unknown = high risk
        if (val >= 100) return 1.0;
        if (val >= 80) return 0.7;
        if (val >= 50) return 0.4;
        if (val >= 0) return 0.1;
    }
    return 0.5;
}

double RiskAwarePlanner::computeCellEnergy(int x, int y, const nav_msgs::msg::OccupancyGrid& costmap,
                                           const std::vector<std::vector<double>>* slope_map,
                                           const std::vector<std::vector<double>>* slip_map) const {
    double energy = config_.base_energy_per_meter;

    // Slope energy from slope map
    if (slope_map && y < static_cast<int>(slope_map->size()) && x < static_cast<int>((*slope_map)[0].size())) {
        double slope_deg = (*slope_map)[y][x];
        if (slope_deg > 0) {
            // Uphill costs more energy (exponential with slope)
            energy += config_.slope_energy_factor * slope_deg * slope_deg / 90.0;
        }
    } else {
        // Estimate from costmap
        int idx = y * costmap.info.width + x;
        if (idx >= 0 && idx < static_cast<int>(costmap.data.size())) {
            int8_t val = costmap.data[idx];
            if (val >= 50 && val < 100) {
                double slope_est = (val - 50) / 50.0 * 20.0; // Rough slope estimate
                energy += config_.slope_energy_factor * slope_est * slope_est / 90.0;
            }
        }
    }

    // Slip energy from slip map
    if (slip_map && y < static_cast<int>(slip_map->size()) && x < static_cast<int>((*slip_map)[0].size())) {
        double slip_prob = (*slip_map)[y][x];
        energy += config_.slip_energy_factor * slip_prob;
    } else {
        // Estimate from costmap risk
        double risk = computeCellRisk(x, y, costmap, nullptr);
        if (risk > 0.5) {
            energy += config_.slip_energy_factor * risk;
        }
    }

    return energy;
}

bool RiskAwarePlanner::isValid(int x, int y, const nav_msgs::msg::OccupancyGrid& costmap) const {
    if (x < 0 || x >= costmap.info.width || y < 0 || y >= costmap.info.height) return false;
    int idx = y * costmap.info.width + x;
    if (idx < 0 || idx >= static_cast<int>(costmap.data.size())) return false;
    int8_t val = costmap.data[idx];
    // Unknown (-1) is NOT valid - treat as high risk / impassable
    if (val == -1) return false;
    // Occupied (100) is not valid
    return val < 100;
}

std::vector<std::pair<int, int>> RiskAwarePlanner::getNeighbors(int x, int y) const {
    std::vector<std::pair<int, int>> neighbors;
    static const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    static const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    int n_dirs = config_.allow_diagonal ? 8 : 4;
    for (int i = 0; i < n_dirs; ++i) {
        neighbors.emplace_back(x + dx[i], y + dy[i]);
    }
    return neighbors;
}

nav_msgs::msg::Path RiskAwarePlanner::reconstructPath(const std::unordered_map<long, GridCell>& cells,
                                                      int goal_x, int goal_y,
                                                      const nav_msgs::msg::MapMetaData& info,
                                                      const geometry_msgs::msg::Pose& start) const {
    nav_msgs::msg::Path path_msg;
    path_msg.header.frame_id = "odom";

    std::vector<std::pair<int, int>> path_indices;
    int cx = goal_x, cy = goal_y;
    while (cx != -1 && cy != -1) {
        path_indices.emplace_back(cx, cy);
        long h = hash(cx, cy);
        auto it = cells.find(h);
        if (it == cells.end()) break;
        cx = it->second.parent_x;
        cy = it->second.parent_y;
    }
    std::reverse(path_indices.begin(), path_indices.end());

    geometry_msgs::msg::PoseStamped start_ps;
    start_ps.header = path_msg.header;
    start_ps.pose = start;
    path_msg.poses.push_back(start_ps);

    for (auto [gx, gy] : path_indices) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = path_msg.header;
        ps.pose.position.x = gridToWorldX(gx, info);
        ps.pose.position.y = gridToWorldY(gy, info);
        ps.pose.position.z = 0.0;
        ps.pose.orientation.w = 1.0;
        path_msg.poses.push_back(ps);
    }

    return path_msg;
}

}  // namespace lunar_slam