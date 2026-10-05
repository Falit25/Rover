#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <vector>
#include <queue>
#include <mutex>

namespace lunar_slam {

struct ExplorationConfig {
    double min_frontier_size = 5;
    double max_frontier_distance = 20.0;
    double safety_margin = 1.0;
    double information_gain_weight = 1.0;
    double distance_weight = 0.5;
    double risk_weight = 2.0;
    double replan_rate = 0.5;
};

struct Frontier {
    std::vector<geometry_msgs::msg::Point> cells;
    geometry_msgs::msg::Point centroid;
    double size = 0.0;
    double information_gain = 0.0;
    double distance = 0.0;
    double risk = 0.0;
    double score = 0.0;
};

class ExplorationManagerNode : public rclcpp::Node {
public:
    explicit ExplorationManagerNode(const rclcpp::NodeOptions& options);

private:
    void costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void timerCallback();
    void updateRobotPosition();
    void detectFrontiers();
    void scoreFrontiers();
    void publishBestFrontier();
    std::vector<Frontier> extractFrontiers(const nav_msgs::msg::OccupancyGrid& grid);
    double computeInformationGain(const Frontier& frontier, const nav_msgs::msg::OccupancyGrid& grid);
    double computeFrontierRisk(const Frontier& frontier, const nav_msgs::msg::OccupancyGrid& grid);
    geometry_msgs::msg::Point gridToWorld(int x, int y, const nav_msgs::msg::MapMetaData& info);
    bool isFrontierCell(int x, int y, const nav_msgs::msg::OccupancyGrid& grid);

    ExplorationConfig config_;
    
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    nav_msgs::msg::OccupancyGrid::SharedPtr latest_costmap_;
    std::mutex data_mutex_;
    geometry_msgs::msg::Point robot_position_;
    std::vector<Frontier> frontiers_;
    std::string base_frame_;
    std::string odom_frame_;
};

}  // namespace lunar_slam