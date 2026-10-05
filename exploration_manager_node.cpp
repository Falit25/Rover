#include "lunar_slam/exploration_manager_node.hpp"
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>
#include <algorithm>
#include <limits>

namespace lunar_slam {

ExplorationManagerNode::ExplorationManagerNode(const rclcpp::NodeOptions& options)
    : Node("exploration_manager_node", options) {

    this->declare_parameter("costmap_topic", "/traversability_costmap");
    this->declare_parameter("goal_topic", "/goal_pose");
    this->declare_parameter("timer_rate", 0.5);
    this->declare_parameter("robot_base_frame", "base_link");
    this->declare_parameter("odom_frame", "odom");

    config_.min_frontier_size = this->declare_parameter("exploration.min_frontier_size", 5);
    config_.max_frontier_distance = this->declare_parameter("exploration.max_frontier_distance", 20.0);
    config_.safety_margin = this->declare_parameter("exploration.safety_margin", 1.0);
    config_.information_gain_weight = this->declare_parameter("exploration.information_gain_weight", 1.0);
    config_.distance_weight = this->declare_parameter("exploration.distance_weight", 0.5);
    config_.risk_weight = this->declare_parameter("exploration.risk_weight", 2.0);

    std::string costmap_topic = this->get_parameter("costmap_topic").as_string();
    std::string goal_topic = this->get_parameter("goal_topic").as_string();
    double timer_rate = this->get_parameter("timer_rate").as_double();
    base_frame_ = this->get_parameter("robot_base_frame").as_string();
    odom_frame_ = this->get_parameter("odom_frame").as_string();

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    costmap_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        costmap_topic, 10, std::bind(&ExplorationManagerNode::costmapCallback, this, std::placeholders::_1));

    goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(goal_topic, 10);

    timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / timer_rate),
        std::bind(&ExplorationManagerNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "Exploration Manager Node initialized");
}

void ExplorationManagerNode::costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_costmap_ = msg;
}

void ExplorationManagerNode::timerCallback() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    if (!latest_costmap_) return;
    
    // Update robot position from TF
    updateRobotPosition();
    
    detectFrontiers();
    scoreFrontiers();
    publishBestFrontier();
}

void ExplorationManagerNode::updateRobotPosition() {
    try {
        geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
            odom_frame_, base_frame_, tf2::TimePointZero);
        robot_position_.x = transform.transform.translation.x;
        robot_position_.y = transform.transform.translation.y;
        robot_position_.z = transform.transform.translation.z;
    } catch (tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "TF lookup failed for robot position: %s", ex.what());
    }
}

void ExplorationManagerNode::detectFrontiers() {
    if (!latest_costmap_) return;
    
    frontiers_ = extractFrontiers(*latest_costmap_);
}

std::vector<Frontier> ExplorationManagerNode::extractFrontiers(const nav_msgs::msg::OccupancyGrid& grid) {
    std::vector<Frontier> frontiers;
    int width = grid.info.width;
    int height = grid.info.height;
    std::vector<bool> visited(width * height, false);
    
    auto idx = [width](int x, int y) { return y * width + x; };
    
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            if (visited[idx(x, y)]) continue;
            if (!isFrontierCell(x, y, grid)) continue;
            
            Frontier frontier;
            std::queue<std::pair<int, int>> q;
            q.emplace(x, y);
            visited[idx(x, y)] = true;
            
            while (!q.empty()) {
                auto [cx, cy] = q.front();
                q.pop();
                
                frontier.cells.push_back(gridToWorld(cx, cy, grid.info));
                frontier.size += 1.0;
                
                static const int dx[4] = {1, -1, 0, 0};
                static const int dy[4] = {0, 0, 1, -1};
                
                for (int i = 0; i < 4; ++i) {
                    int nx = cx + dx[i];
                    int ny = cy + dy[i];
                    if (nx >= 1 && nx < width - 1 && ny >= 1 && ny < height - 1 && !visited[idx(nx, ny)]) {
                        if (isFrontierCell(nx, ny, grid)) {
                            visited[idx(nx, ny)] = true;
                            q.emplace(nx, ny);
                        }
                    }
                }
            }
            
            if (frontier.size >= config_.min_frontier_size) {
                double sum_x = 0, sum_y = 0;
                for (const auto& cell : frontier.cells) {
                    sum_x += cell.x;
                    sum_y += cell.y;
                }
                frontier.centroid.x = sum_x / frontier.cells.size();
                frontier.centroid.y = sum_y / frontier.cells.size();
                frontier.centroid.z = 0.0;
                frontiers.push_back(std::move(frontier));
            }
        }
    }
    
    return frontiers;
}

bool ExplorationManagerNode::isFrontierCell(int x, int y, const nav_msgs::msg::OccupancyGrid& grid) {
    int idx = y * grid.info.width + x;
    int8_t cell = grid.data[idx];
    if (cell != 0 && cell != -1) return false;
    
    static const int dx[4] = {1, -1, 0, 0};
    static const int dy[4] = {0, 0, 1, -1};
    
    for (int i = 0; i < 4; ++i) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        int nidx = ny * grid.info.width + nx;
        int8_t ncell = grid.data[nidx];
        if (ncell == -1) return true;
    }
    return false;
}

geometry_msgs::msg::Point ExplorationManagerNode::gridToWorld(int x, int y, const nav_msgs::msg::MapMetaData& info) {
    geometry_msgs::msg::Point p;
    p.x = info.origin.position.x + (x + 0.5) * info.resolution;
    p.y = info.origin.position.y + (y + 0.5) * info.resolution;
    p.z = 0.0;
    return p;
}

void ExplorationManagerNode::scoreFrontiers() {
    for (auto& frontier : frontiers_) {
        frontier.information_gain = computeInformationGain(frontier, *latest_costmap_);
        
        frontier.distance = std::hypot(
            frontier.centroid.x - robot_position_.x,
            frontier.centroid.y - robot_position_.y
        );
        
        frontier.risk = computeFrontierRisk(frontier, *latest_costmap_);
        
        if (frontier.distance > config_.max_frontier_distance) {
            frontier.score = -std::numeric_limits<double>::infinity();
            continue;
        }
        
        frontier.score = config_.information_gain_weight * frontier.information_gain
                       - config_.distance_weight * frontier.distance
                       - config_.risk_weight * frontier.risk;
    }
}

double ExplorationManagerNode::computeInformationGain(const Frontier& frontier, const nav_msgs::msg::OccupancyGrid& grid) {
    return frontier.size * grid.info.resolution * grid.info.resolution;
}

double ExplorationManagerNode::computeFrontierRisk(const Frontier& frontier, const nav_msgs::msg::OccupancyGrid& grid) {
    double risk_sum = 0.0;
    int count = 0;
    double res = grid.info.resolution;
    double origin_x = grid.info.origin.position.x;
    double origin_y = grid.info.origin.position.y;
    
    for (const auto& cell : frontier.cells) {
        int gx = static_cast<int>((cell.x - origin_x) / res);
        int gy = static_cast<int>((cell.y - origin_y) / res);
        
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                int nx = gx + dx;
                int ny = gy + dy;
                if (nx >= 0 && nx < grid.info.width && ny >= 0 && ny < grid.info.height) {
                    int idx = ny * grid.info.width + nx;
                    int8_t val = grid.data[idx];
                    if (val > 0) risk_sum += val / 100.0;
                    count++;
                }
            }
        }
    }
    
    return count > 0 ? risk_sum / count : 0.0;
}

void ExplorationManagerNode::publishBestFrontier() {
    if (frontiers_.empty()) return;
    
    auto best_it = std::max_element(frontiers_.begin(), frontiers_.end(),
        [](const Frontier& a, const Frontier& b) { return a.score < b.score; });
    
    if (best_it->score <= -std::numeric_limits<double>::infinity()/2) return;
    
    geometry_msgs::msg::PoseStamped goal;
    goal.header.frame_id = "odom";
    goal.header.stamp = this->now();
    goal.pose.position = best_it->centroid;
    goal.pose.orientation.w = 1.0;
    
    goal_pub_->publish(goal);
    
    RCLCPP_INFO(this->get_logger(), "Published exploration goal: (%.2f, %.2f), score: %.2f",
                best_it->centroid.x, best_it->centroid.y, best_it->score);
}

}  // namespace lunar_slam

RCLCPP_COMPONENTS_REGISTER_NODE(lunar_slam::ExplorationManagerNode)