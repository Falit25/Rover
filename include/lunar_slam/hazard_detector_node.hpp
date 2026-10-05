#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include "lunar_slam/hazard_detector.hpp"
#include "lunar_slam/mobility_risk_model.hpp"

namespace lunar_slam {

class HazardDetectorNode : public rclcpp::Node {
public:
    explicit HazardDetectorNode(const rclcpp::NodeOptions& options);

private:
    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void thermalCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void fusedOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void timerCallback();
    void enhanceCostmapWithMobilityRisk(nav_msgs::msg::OccupancyGrid& grid, 
                                         const geometry_msgs::msg::TransformStamped& odom_to_base);
    nav_msgs::msg::OccupancyGrid generateSlopeMap(const sensor_msgs::msg::PointCloud2& cloud_msg);
    nav_msgs::msg::OccupancyGrid generateSurfaceRiskMap(const sensor_msgs::msg::PointCloud2& cloud_msg);

    HazardConfig config_;
    std::unique_ptr<HazardDetector> hazard_detector_;
    std::unique_ptr<MobilityRiskModel> mobility_risk_model_;
    
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr thermal_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr fused_odom_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr hazard_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr slope_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr surface_risk_map_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    cv::Mat latest_thermal_;
    sensor_msgs::msg::CameraInfo latest_camera_info_;
    bool have_camera_info_ = false;
    std::mutex thermal_mutex_;
    std::string lidar_frame_;
    std::string thermal_frame_;
    std::string base_frame_;
    std::string odom_frame_;
    
    nav_msgs::msg::Odometry::SharedPtr latest_fused_odom_;
    std::mutex fused_odom_mutex_;
    
    bool use_mobility_risk_ = true;
    double mobility_risk_weight_ = 0.7;
    double hazard_weight_ = 0.3;
    double uncertainty_weight_ = 0.2;  // NEW: weight for uncertainty penalty
};

}  // namespace lunar_slam