#include "lunar_slam/risk_aware_planner_node.hpp"
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>
#include <limits>

namespace lunar_slam {

RiskAwarePlannerNode::RiskAwarePlannerNode(const rclcpp::NodeOptions& options)
    : Node("risk_aware_planner_node", options) {

    this->declare_parameter("costmap_topic", "/traversability_costmap");
    this->declare_parameter("goal_topic", "/goal_pose");
    this->declare_parameter("global_path_topic", "/global_path");
    this->declare_parameter("timer_rate", 2.0);
    this->declare_parameter("robot_base_frame", "base_link");
    this->declare_parameter("odom_frame", "odom");
    this->declare_parameter("slope_map_topic", "/slope_map");
    this->declare_parameter("surface_risk_map_topic", "/surface_risk_map");

    config_.risk_weight = this->declare_parameter("risk_aware_planner.risk_weight", 2.0);
    config_.max_risk_per_step = this->declare_parameter("risk_aware_planner.max_risk_per_step", 0.3);
    config_.risk_accumulation_threshold = this->declare_parameter("risk_aware_planner.risk_accumulation_threshold", 0.95);
    config_.max_iterations = this->declare_parameter("risk_aware_planner.max_iterations", 10000);
    config_.heuristic_weight = this->declare_parameter("risk_aware_planner.heuristic_weight", 1.0);
    config_.allow_diagonal = this->declare_parameter("risk_aware_planner.allow_diagonal", true);
    config_.diagonal_cost = this->declare_parameter("risk_aware_planner.diagonal_cost", 1.414);
    
    // Energy-aware parameters
    config_.energy_weight = this->declare_parameter("risk_aware_planner.energy_weight", 1.5);
    config_.slope_energy_factor = this->declare_parameter("risk_aware_planner.slope_energy_factor", 2.0);
    config_.slip_energy_factor = this->declare_parameter("risk_aware_planner.slip_energy_factor", 3.0);
    config_.base_energy_per_meter = this->declare_parameter("risk_aware_planner.base_energy_per_meter", 1.0);
    config_.max_energy_per_step = this->declare_parameter("risk_aware_planner.max_energy_per_step", 50.0);
    config_.battery_capacity = this->declare_parameter("risk_aware_planner.battery_capacity", 1000.0);
    config_.enable_energy_planning = this->declare_parameter("risk_aware_planner.enable_energy_planning", true);

    planner_ = std::make_unique<RiskAwarePlanner>(config_);

    std::string costmap_topic = this->get_parameter("costmap_topic").as_string();
    std::string goal_topic = this->get_parameter("goal_topic").as_string();
    std::string global_path_topic = this->get_parameter("global_path_topic").as_string();
    std::string slope_map_topic = this->get_parameter("slope_map_topic").as_string();
    std::string surface_risk_map_topic = this->get_parameter("surface_risk_map_topic").as_string();
    double timer_rate = this->get_parameter("timer_rate").as_double();
    base_frame_ = this->get_parameter("robot_base_frame").as_string();
    odom_frame_ = this->get_parameter("odom_frame").as_string();

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    costmap_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        costmap_topic, 10, std::bind(&RiskAwarePlannerNode::costmapCallback, this, std::placeholders::_1));

    slope_map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        slope_map_topic, 10, std::bind(&RiskAwarePlannerNode::slopeMapCallback, this, std::placeholders::_1));

    surface_risk_map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        surface_risk_map_topic, 10, std::bind(&RiskAwarePlannerNode::surfaceRiskMapCallback, this, std::placeholders::_1));

    goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        goal_topic, 10, std::bind(&RiskAwarePlannerNode::goalCallback, this, std::placeholders::_1));

    path_pub_ = this->create_publisher<nav_msgs::msg::Path>(global_path_topic, 10);

    // Active probing publisher
    probing_goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/probing_goal", 10);

    timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / timer_rate),
        std::bind(&RiskAwarePlannerNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "Risk-Aware Planner Node initialized with Energy-Aware Planning");
}

void RiskAwarePlannerNode::costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_costmap_ = msg;
    have_costmap_ = true;
}

void RiskAwarePlannerNode::slopeMapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_slope_map_ = msg;
    have_slope_map_ = true;
}

void RiskAwarePlannerNode::surfaceRiskMapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_surface_risk_map_ = msg;
    have_surface_risk_map_ = true;
}

void RiskAwarePlannerNode::goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_goal_ = msg;
    have_goal_ = true;
}

void RiskAwarePlannerNode::timerCallback() {
    planAndPublish();
    checkAndPublishProbingGoal();
}

geometry_msgs::msg::Pose RiskAwarePlannerNode::getRobotPose() {
    geometry_msgs::msg::Pose pose;
    try {
        geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
            odom_frame_, base_frame_, tf2::TimePointZero);
        pose.position.x = transform.transform.translation.x;
        pose.position.y = transform.transform.translation.y;
        pose.position.z = transform.transform.translation.z;
        pose.orientation = transform.transform.rotation;
    } catch (tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "TF lookup failed: %s", ex.what());
        pose.position.x = pose.position.y = pose.position.z = 0.0;
        pose.orientation.w = 1.0;
    }
    return pose;
}

void RiskAwarePlannerNode::planAndPublish() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    if (!have_costmap_ || !have_goal_ || !latest_costmap_ || !latest_goal_) {
        return;
    }

    geometry_msgs::msg::Pose start_pose = getRobotPose();

    // Convert slope and surface risk maps to vector format
    std::vector<std::vector<double>> slope_map_vec, surface_risk_map_vec;
    if (have_slope_map_ && latest_slope_map_) {
        slope_map_vec = occupancyGridToVector(*latest_slope_map_);
    }
    if (have_surface_risk_map_ && latest_surface_risk_map_) {
        surface_risk_map_vec = occupancyGridToVector(*latest_surface_risk_map_);
    }

    auto path = planner_->planPath(*latest_costmap_, start_pose, latest_goal_->pose,
                                   nullptr, 
                                   have_slope_map_ ? &slope_map_vec : nullptr,
                                   have_surface_risk_map_ ? &surface_risk_map_vec : nullptr);
    
    if (!path.poses.empty()) {
        path.header = latest_costmap_->header;
        path_pub_->publish(path);
        
        // Log path energy
        double total_energy = computePathEnergy(path);
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
            "Path published: %zu waypoints, est. energy: %.1f J", path.poses.size(), total_energy);
    }
}

void RiskAwarePlannerNode::checkAndPublishProbingGoal() {
    if (!config_.enable_energy_planning || !have_costmap_ || !latest_costmap_) return;
    
    // Check for high uncertainty regions along planned path
    // If uncertainty > threshold, publish probing goal
    double uncertainty_threshold = this->get_parameter("active_probing.uncertainty_threshold").as_double();
    
    // Find cells with high uncertainty (unknown cells or high model uncertainty)
    if (latest_costmap_) {
        const auto& grid = *latest_costmap_;
        int width = grid.info.width;
        int height = grid.info.height;
        
        // Check region around robot
        geometry_msgs::msg::Pose robot_pose = getRobotPose();
        int robot_x = worldToGridX(robot_pose.position.x, grid.info);
        int robot_y = worldToGridY(robot_pose.position.y, grid.info);
        
        int search_radius = static_cast<int>(this->get_parameter("active_probing.search_radius_m").as_double() / grid.info.resolution);
        double min_probe_dist = this->get_parameter("active_probing.min_probe_distance_m").as_double();
        double max_probe_dist = this->get_parameter("active_probing.max_probe_distance_m").as_double();
        
        int min_radius = static_cast<int>(min_probe_dist / grid.info.resolution);
        int max_radius = static_cast<int>(max_probe_dist / grid.info.resolution);
        
        double max_uncertainty = 0.0;
        int best_x = -1, best_y = -1;
        
        for (int dy = -search_radius; dy <= search_radius; ++dy) {
            for (int dx = -search_radius; dx <= search_radius; ++dx) {
                int x = robot_x + dx;
                int y = robot_y + dy;
                if (x < 0 || x >= width || y < 0 || y >= height) continue;
                
                int idx = y * width + x;
                int8_t val = grid.data[idx];
                
                // Compute uncertainty: unknown cells have high uncertainty
                // Known cells with high cost (hazard + mobility risk) also have uncertainty
                double cell_uncertainty = 0.0;
                
                if (val == -1) {
                    // Unknown terrain = maximum uncertainty
                    cell_uncertainty = 1.0;
                } else if (val > 50) {
                    // High cost cells have elevated uncertainty
                    cell_uncertainty = val / 100.0;
                }
                
                // Distance from robot
                double dist = std::hypot(dx, dy) * grid.info.resolution;
                
                // Only consider cells within probe distance range
                if (dist > min_probe_dist && dist < max_probe_dist) {
                    // Score = uncertainty * (1 / distance) - prioritize uncertain but reachable cells
                    double score = cell_uncertainty / (dist + 0.1);
                    if (score > max_uncertainty) {
                        max_uncertainty = score;
                        best_x = x;
                        best_y = y;
                    }
                }
            }
        }
        
        if (best_x >= 0 && max_uncertainty > uncertainty_threshold) {
            geometry_msgs::msg::PoseStamped probing_goal;
            probing_goal.header.frame_id = "odom";
            probing_goal.header.stamp = this->now();
            probing_goal.pose.position.x = gridToWorldX(best_x, grid.info);
            probing_goal.pose.position.y = gridToWorldY(best_y, grid.info);
            probing_goal.pose.position.z = 0.0;
            probing_goal.pose.orientation.w = 1.0;
            
            probing_goal_pub_->publish(probing_goal);
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 30000,
                "Active probing goal published: (%.2f, %.2f), uncertainty_score: %.2f", 
                probing_goal.pose.position.x, probing_goal.pose.position.y, max_uncertainty);
        }
    }
}

double RiskAwarePlannerNode::computePathEnergy(const nav_msgs::msg::Path& path) const {
    double energy = 0.0;
    for (size_t i = 1; i < path.poses.size(); ++i) {
        double dx = path.poses[i].pose.position.x - path.poses[i-1].pose.position.x;
        double dy = path.poses[i].pose.position.y - path.poses[i-1].pose.position.y;
        double dist = std::hypot(dx, dy);
        energy += config_.base_energy_per_meter * dist;
    }
    return energy;
}

std::vector<std::vector<double>> RiskAwarePlannerNode::occupancyGridToVector(
    const nav_msgs::msg::OccupancyGrid& grid) {
    std::vector<std::vector<double>> vec(grid.info.height, 
        std::vector<double>(grid.info.width, 0.0));
    
    for (int y = 0; y < grid.info.height; ++y) {
        for (int x = 0; x < grid.info.width; ++x) {
            int idx = y * grid.info.width + x;
            int8_t val = grid.data[idx];
            if (val == -1) {
                vec[y][x] = -1.0;
            } else {
                vec[y][x] = val / 100.0; // Normalize to 0-1
            }
        }
    }
    return vec;
}

int RiskAwarePlannerNode::worldToGridX(double wx, const nav_msgs::msg::MapMetaData& info) const {
    return static_cast<int>(std::floor((wx - info.origin.position.x) / info.resolution));
}

int RiskAwarePlannerNode::worldToGridY(double wy, const nav_msgs::msg::MapMetaData& info) const {
    return static_cast<int>(std::floor((wy - info.origin.position.y) / info.resolution));
}

double RiskAwarePlannerNode::gridToWorldX(int gx, const nav_msgs::msg::MapMetaData& info) const {
    return info.origin.position.x + (gx + 0.5) * info.resolution;
}

double RiskAwarePlannerNode::gridToWorldY(int gy, const nav_msgs::msg::MapMetaData& info) const {
    return info.origin.position.y + (gy + 0.5) * info.resolution;
}

}  // namespace lunar_slam

RCLCPP_COMPONENTS_REGISTER_NODE(lunar_slam::RiskAwarePlannerNode)