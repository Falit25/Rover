#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <mutex>

namespace lunar_slam {

class LidarOdometryNode : public rclcpp::Node {
public:
    explicit LidarOdometryNode(const rclcpp::NodeOptions& options);

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void publishOdom(const Eigen::Matrix4d& pose, const rclcpp::Time& stamp);

    struct Config {
        double max_correspondence_distance = 1.0;
        double transformation_epsilon = 1e-6;
        double euclidean_fitness_epsilon = 1e-6;
        int max_iterations = 30;
        double voxel_leaf_size = 0.1;
        double min_overlap = 0.3;
    };

    Config config_;
    
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr prev_cloud_;
    Eigen::Matrix4d last_pose_ = Eigen::Matrix4d::Identity();
    bool initialized_ = false;
    std::string base_frame_, odom_frame_, lidar_frame_;
    bool publish_tf_;
    std::mutex cloud_mutex_;
};

}  // namespace lunar_slam