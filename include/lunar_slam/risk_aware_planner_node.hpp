#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include "lunar_slam/risk_aware_planner.hpp"

namespace lunar_slam {

class RiskAwarePlannerNode : public rclcpp::Node {
public:
    explicit RiskAwarePlannerNode(const rclcpp::NodeOptions& options);

private:
    void costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void slopeMapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void surfaceRiskMapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void timerCallback();
    void planAndPublish();
    void checkAndPublishProbingGoal();
    geometry_msgs::msg::Pose getRobotPose();
    double computePathEnergy(const nav_msgs::msg::Path& path) const;
    std::vector<std::vector<double>> occupancyGridToVector(const nav_msgs::msg::OccupancyGrid& grid);
    int worldToGridX(double wx, const nav_msgs::msg::MapMetaData& info) const;
    int worldToGridY(double wy, const nav_msgs::msg::MapMetaData& info) const;
    double gridToWorldX(int gx, const nav_msgs::msg::MapMetaData& info) const;
    double gridToWorldY(int gy, const nav_msgs::msg::MapMetaData& info) const;

    RiskAwareConfig config_;
    std::unique_ptr<RiskAwarePlanner> planner_;
    
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr slope_map_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr surface_risk_map_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr probing_goal_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    nav_msgs::msg::OccupancyGrid::SharedPtr latest_costmap_;
    nav_msgs::msg::OccupancyGrid::SharedPtr latest_slope_map_;
    nav_msgs::msg::OccupancyGrid::SharedPtr latest_surface_risk_map_;
    geometry_msgs::msg::PoseStamped::SharedPtr latest_goal_;
    bool have_costmap_ = false;
    bool have_slope_map_ = false;
    bool have_surface_risk_map_ = false;
    bool have_goal_ = false;
    std::mutex data_mutex_;
    std::string base_frame_;
    std::string odom_frame_;
};

}  // namespace lunar_slam