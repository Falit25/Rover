#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <mutex>
#include <deque>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "lunar_slam/mobility_risk_model.hpp"

namespace lunar_slam {

class MobilityDatasetCollector : public rclcpp::Node {
public:
    explicit MobilityDatasetCollector(const rclcpp::NodeOptions& options);

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void thermalCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void groundTruthCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void wheelOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void fusedOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void elevationMapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void timerCallback();
    
    void processAndRecord();
    double computeWheelSlipFromGazebo() const;
    double computeSinkage() const;
    double getTerrainHeight(double x, double y) const;
    TerrainFeatures extractFeaturesAtRobot() const;
    MobilityRiskPrediction createGroundTruth(double slip, double sinkage) const;
    void writeToCSV(const TerrainFeatures& features, const MobilityRiskPrediction& gt);
    void flushBuffer();
    TerrainFeatures extractFeaturesFromLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) const;

    struct Config {
        std::string output_file = "mobility_dataset.csv";
        double slip_threshold = 0.15;
        double sinkage_threshold = 0.05;
        double record_rate = 2.0;  // Hz
        int buffer_size = 100;
        bool record_thermal = true;
        std::string robot_base_frame = "base_link";
        std::string odom_frame = "odom";
        std::string gazebo_ground_truth_topic = "/ground_truth/odom";
        std::vector<std::string> wheel_joint_names = {"rear_left_wheel_joint", "rear_right_wheel_joint"};
        double wheel_radius = 0.15;
        double elevation_map_resolution = 0.1;  // 10cm grid
    };
    
    Config config_;
    
    // Sensor buffers
    nav_msgs::msg::Odometry::SharedPtr latest_ground_truth_;
    nav_msgs::msg::Odometry::SharedPtr latest_wheel_odom_;
    nav_msgs::msg::Odometry::SharedPtr latest_fused_odom_;
    geometry_msgs::msg::Twist::SharedPtr latest_cmd_vel_;
    nav_msgs::msg::OccupancyGrid::SharedPtr latest_costmap_;
    nav_msgs::msg::OccupancyGrid::SharedPtr latest_elevation_map_;
    
    std::mutex buffer_mutex_;
    
    // Previous state for slip computation
    sensor_msgs::msg::JointState::SharedPtr prev_joint_state_;
    nav_msgs::msg::Odometry::SharedPtr prev_ground_truth_;
    rclcpp::Time prev_time_;
    
    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // Output
    std::ofstream csv_file_;
    std::vector<std::pair<TerrainFeatures, MobilityRiskPrediction>> write_buffer_;
    
    rclcpp::TimerBase::SharedPtr timer_;
    
    // Frame names
    std::string base_frame_ = "base_link";
    std::string odom_frame_ = "odom";
};

}  // namespace lunar_slam