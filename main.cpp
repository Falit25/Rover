#include <iostream>
#include <memory>
#include <vector>

// ROS 2 Core Client Library
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

// OpenCV
#include <opencv2/opencv.hpp>

// Project Utilities
#include "lunar_slam/math_utils.hpp"
#include "lunar_slam/traversability_mapper.hpp"
#include "lunar_slam/thermal_processing.hpp"
#include "lunar_slam/path_planner.hpp"

class LunarSlamNode : public rclcpp::Node {
public:
    LunarSlamNode() : Node("lunar_slam_node") {
        RCLCPP_INFO(this->get_logger(), "Initializing Lunar SLAM ROS 2 Pipeline...");

        // Initialize Internal Pose State
        current_pose_ = Eigen::Matrix4d::Identity();
        current_orientation_ = Eigen::Quaterniond::Identity();

        // Initialize SLAM Modules
        traversability_mapper_ = std::make_unique<lunar_slam::TraversabilityMapper>(0.1, 200, 0.15);
        thermal_processor_ = std::make_unique<lunar_slam::ThermalProcessor>();
        path_planner_ = std::make_unique<lunar_slam::PathPlanner>();

        // 1. SENSOR DATA INPUT SUBSCRIBERS
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu/data", 10,
            std::bind(&LunarSlamNode::imuCallback, this, std::placeholders::_1));

        lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 10,
            std::bind(&LunarSlamNode::lidarCallback, this, std::placeholders::_1));

        // 3. POSE PUBLISHERS & TF BROADCASTER
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        // 4. TRAVERSABILITY & PLANNING PUBLISHERS
        costmap_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/traversability_costmap", 10);
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/planned_path", 10);

        // 5. THERMAL IMAGE SUBSCRIBER
        thermal_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/thermal/image_raw", 10,
            std::bind(&LunarSlamNode::thermalCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Subscribers and Publishers configured successfully.");
    }

private:
    // --- 1. SENSOR DATA INPUT CALLBACKS ---

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        Eigen::Vector3d delta_omega(
            msg->angular_velocity.x * 0.01, // assuming dt = 0.01s
            msg->angular_velocity.y * 0.01,
            msg->angular_velocity.z * 0.01
        );

        // Apply Lie Algebra update via math_utils
        current_orientation_ = lunar_slam::updateQuaternion(current_orientation_, delta_omega);
    }

    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // --- 2. POINT-CLOUD ALIGNMENT (ICP SIMULATION) ---
        Eigen::Matrix3d delta_R = lunar_slam::exp_so3(Eigen::Vector3d(0.0, 0.0, 0.01));
        Eigen::Vector3d delta_t(0.05, 0.0, 0.0); // 5cm forward motion per scan

        // Combine into SE(3) transformation matrix via math_utils
        Eigen::Matrix4d T_delta = lunar_slam::createSE3Matrix(delta_R, delta_t);

        // Update overall robot trajectory pose frame
        current_pose_ = current_pose_ * T_delta;

        // Generate traversability costmap
        auto costmap = traversability_mapper_->generateCostmap(msg);
        costmap_pub_->publish(costmap);

        // Plan path from current pose to a fixed goal (10m forward)
        geometry_msgs::msg::Pose start_pose;
        start_pose.position.x = current_pose_(0, 3);
        start_pose.position.y = current_pose_(1, 3);
        start_pose.position.z = current_pose_(2, 3);
        start_pose.orientation.w = current_orientation_.w();
        start_pose.orientation.x = current_orientation_.x();
        start_pose.orientation.y = current_orientation_.y();
        start_pose.orientation.z = current_orientation_.z();

        geometry_msgs::msg::Pose goal_pose = start_pose;
        goal_pose.position.x += 10.0; // 10m forward goal

        auto path = path_planner_->planSafePath(costmap, start_pose, goal_pose);
        path_pub_->publish(path);

        // Broadcast updated state to ROS 2 ecosystem
        publishPoseAndTF(msg->header.stamp);
    }

    void thermalCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        // Convert ROS Image to OpenCV Mat (assuming mono16 thermal)
        cv::Mat raw_thermal(msg->height, msg->width, CV_16UC1, const_cast<uint8_t*>(msg->data.data()), msg->step);
        
        cv::Mat normalized = thermal_processor_->normalize16To8Bit(raw_thermal);
        cv::Mat enhanced = thermal_processor_->enhanceThermalFrame(normalized);
        
        // Store for feature matching (would need previous frame buffer in real implementation)
        last_thermal_frame_ = enhanced.clone();
    }

    // --- 3. POSE PUBLISHER & TF BROADCASTING ---

    void publishPoseAndTF(const rclcpp::Time& stamp) {
        Eigen::Vector3d translation = current_pose_.block<3, 1>(0, 3);

        // A. Publish /odom topic for Nav2
        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.stamp = stamp;
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "base_link";

        odom_msg.pose.pose.position.x = translation.x();
        odom_msg.pose.pose.position.y = translation.y();
        odom_msg.pose.pose.position.z = translation.z();

        odom_msg.pose.pose.orientation.w = current_orientation_.w();
        odom_msg.pose.pose.orientation.x = current_orientation_.x();
        odom_msg.pose.pose.orientation.y = current_orientation_.y();
        odom_msg.pose.pose.orientation.z = current_orientation_.z();

        odom_pub_->publish(odom_msg);

        // B. Broadcast Transform to /tf for RViz2 Visualization
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = stamp;
        tf_msg.header.frame_id = "odom";
        tf_msg.child_frame_id = "base_link";

        tf_msg.transform.translation.x = translation.x();
        tf_msg.transform.translation.y = translation.y();
        tf_msg.transform.translation.z = translation.z();

        tf_msg.transform.rotation.w = current_orientation_.w();
        tf_msg.transform.rotation.x = current_orientation_.x();
        tf_msg.transform.rotation.y = current_orientation_.y();
        tf_msg.transform.rotation.z = current_orientation_.z();

        tf_broadcaster_->sendTransform(tf_msg);
    }

    // Member Variables
    Eigen::Matrix4d current_pose_;
    Eigen::Quaterniond current_orientation_;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr thermal_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    std::unique_ptr<lunar_slam::TraversabilityMapper> traversability_mapper_;
    std::unique_ptr<lunar_slam::ThermalProcessor> thermal_processor_;
    std::unique_ptr<lunar_slam::PathPlanner> path_planner_;
    cv::Mat last_thermal_frame_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    std::cout << "=====================================================\n";
    std::cout << " LUNAR SLAM: ROS 2 Node Architecture Loaded          \n";
    std::cout << "=====================================================\n\n";

    auto node = std::make_shared<LunarSlamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}