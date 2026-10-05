#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <vector>
#include <mutex>

namespace lunar_slam {

struct LocalPlannerConfig {
    double max_vel_x = 0.5;
    double min_vel_x = -0.2;
    double max_vel_theta = 1.0;
    double acc_lim_x = 0.5;
    double acc_lim_theta = 1.5;
    double sim_time = 2.0;
    double sim_granularity = 0.05;
    int vx_samples = 10;
    int vtheta_samples = 20;
    double path_distance_bias = 0.6;
    double goal_distance_bias = 0.4;
    double occdist_scale = 0.5;
    double risk_scale = 2.0;
};

struct Trajectory {
    std::vector<geometry_msgs::msg::Pose2D> poses;
    double cost = 0.0;
    bool valid = false;
};

class LocalPlannerNode : public rclcpp::Node {
public:
    explicit LocalPlannerNode(const rclcpp::NodeOptions& options);

private:
    void globalPathCallback(const nav_msgs::msg::Path::SharedPtr msg);
    void costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void timerCallback();
    Trajectory evaluateTrajectory(double vx, double vtheta);
    double computeTrajectoryCost(const Trajectory& traj);
    geometry_msgs::msg::Pose2D getRobotPose();

    LocalPlannerConfig config_;
    
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr global_path_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_path_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    nav_msgs::msg::Path::SharedPtr latest_global_path_;
    nav_msgs::msg::OccupancyGrid::SharedPtr latest_costmap_;
    size_t current_waypoint_idx_ = 0;
    std::mutex data_mutex_;
};

}  // namespace lunar_slam