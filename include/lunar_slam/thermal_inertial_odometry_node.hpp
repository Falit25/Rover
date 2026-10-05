#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include "lunar_slam/thermal_inertial_odometry.hpp"

namespace lunar_slam {

class ThermalInertialOdometryNode : public rclcpp::Node {
public:
    explicit ThermalInertialOdometryNode(const rclcpp::NodeOptions& options);

private:
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void thermalCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void timerCallback();

    ThermalInertialOdometry::Config config_;
    std::unique_ptr<ThermalInertialOdometry> tio_;
    
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr thermal_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::string base_frame_, odom_frame_;
    bool publish_tf_;
};

}  // namespace lunar_slam