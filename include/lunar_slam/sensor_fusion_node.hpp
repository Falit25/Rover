#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include "lunar_slam/sensor_fusion.hpp"

namespace lunar_slam {

class SensorFusionNode : public rclcpp::Node {
public:
    explicit SensorFusionNode(const rclcpp::NodeOptions& options);

private:
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void lidarOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void thermalOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void wheelOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void timerCallback();

    SensorFusionConfig config_;
    std::unique_ptr<SensorFusionEKF> ekf_;
    
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr lidar_odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr thermal_odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_odom_sub_;
    
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr fused_odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr fused_pose_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    rclcpp::TimerBase::SharedPtr timer_;
    std::string base_frame_, odom_frame_;
    bool publish_tf_;
};

}  // namespace lunar_slam