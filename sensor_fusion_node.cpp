#include "lunar_slam/sensor_fusion_node.hpp"
#include <memory>

namespace lunar_slam {

SensorFusionNode::SensorFusionNode(const rclcpp::NodeOptions& options)
    : Node("sensor_fusion_node", options) {

    this->declare_parameter("imu_topic", "/imu/data");
    this->declare_parameter("lidar_odom_topic", "/lidar_odom");
    this->declare_parameter("thermal_odom_topic", "/thermal_inertial_odom");
    this->declare_parameter("wheel_odom_topic", "/wheel_odom");
    this->declare_parameter("fused_odom_topic", "/fused_odom");
    this->declare_parameter("fused_pose_topic", "/fused_pose");
    this->declare_parameter("base_frame", "base_link");
    this->declare_parameter("odom_frame", "odom");
    this->declare_parameter("publish_tf", false);
    this->declare_parameter("timer_rate", 50.0);

    config_.imu_rate = this->declare_parameter("sensor_fusion.imu_rate", 200.0);
    config_.lidar_rate = this->declare_parameter("sensor_fusion.lidar_rate", 10.0);
    config_.thermal_rate = this->declare_parameter("sensor_fusion.thermal_rate", 10.0);
    config_.wheel_rate = this->declare_parameter("sensor_fusion.wheel_rate", 50.0);
    config_.gravity = Eigen::Vector3d(
        this->declare_parameter("sensor_fusion.gravity_x", 0.0),
        this->declare_parameter("sensor_fusion.gravity_y", 0.0),
        this->declare_parameter("sensor_fusion.gravity_z", -1.62)
    );

    std::vector<double> imu_noise_diag = this->declare_parameter("sensor_fusion.imu_noise_diag", 
        std::vector<double>{1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4});
    config_.imu_noise = Eigen::Matrix<double, 6, 6>::Identity();
    config_.imu_noise.diagonal() = Eigen::Map<Eigen::VectorXd>(imu_noise_diag.data(), 6);

    std::vector<double> lidar_noise_diag = this->declare_parameter("sensor_fusion.lidar_noise_diag",
        std::vector<double>{1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2});
    config_.lidar_noise = Eigen::Matrix<double, 6, 6>::Identity();
    config_.lidar_noise.diagonal() = Eigen::Map<Eigen::VectorXd>(lidar_noise_diag.data(), 6);

    std::vector<double> thermal_noise_diag = this->declare_parameter("sensor_fusion.thermal_noise_diag",
        std::vector<double>{5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2});
    config_.thermal_noise = Eigen::Matrix<double, 6, 6>::Identity();
    config_.thermal_noise.diagonal() = Eigen::Map<Eigen::VectorXd>(thermal_noise_diag.data(), 6);

    std::vector<double> wheel_noise_diag = this->declare_parameter("sensor_fusion.wheel_noise_diag",
        std::vector<double>{1e-3, 1e-3, 1e-3});
    config_.wheel_noise = Eigen::Matrix<double, 3, 3>::Identity();
    config_.wheel_noise.diagonal() = Eigen::Map<Eigen::VectorXd>(wheel_noise_diag.data(), 3);

    config_.bias_random_walk_accel = this->declare_parameter("sensor_fusion.bias_random_walk_accel", 1e-5);
    config_.bias_random_walk_gyro = this->declare_parameter("sensor_fusion.bias_random_walk_gyro", 1e-6);

    ekf_ = std::make_unique<SensorFusionEKF>(config_);

    std::string imu_topic = this->get_parameter("imu_topic").as_string();
    std::string lidar_odom_topic = this->get_parameter("lidar_odom_topic").as_string();
    std::string thermal_odom_topic = this->get_parameter("thermal_odom_topic").as_string();
    std::string wheel_odom_topic = this->get_parameter("wheel_odom_topic").as_string();
    std::string fused_odom_topic = this->get_parameter("fused_odom_topic").as_string();
    std::string fused_pose_topic = this->get_parameter("fused_pose_topic").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();
    odom_frame_ = this->get_parameter("odom_frame").as_string();
    publish_tf_ = this->get_parameter("publish_tf").as_bool();
    double timer_rate = this->get_parameter("timer_rate").as_double();

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic, 100, std::bind(&SensorFusionNode::imuCallback, this, std::placeholders::_1));

    lidar_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        lidar_odom_topic, 10, std::bind(&SensorFusionNode::lidarOdomCallback, this, std::placeholders::_1));

    thermal_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        thermal_odom_topic, 10, std::bind(&SensorFusionNode::thermalOdomCallback, this, std::placeholders::_1));

    wheel_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        wheel_odom_topic, 10, std::bind(&SensorFusionNode::wheelOdomCallback, this, std::placeholders::_1));

    fused_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(fused_odom_topic, 10);
    fused_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(fused_pose_topic, 10);

    if (publish_tf_) {
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / timer_rate),
        std::bind(&SensorFusionNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "Sensor Fusion Node initialized");
}

void SensorFusionNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    ekf_->updateIMU(msg);
}

void SensorFusionNode::lidarOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    ekf_->updateLidarOdom(msg);
}

void SensorFusionNode::thermalOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    ekf_->updateThermalOdom(msg);
}

void SensorFusionNode::wheelOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    ekf_->updateWheelOdom(msg);
}

void SensorFusionNode::timerCallback() {
    Eigen::Matrix4d pose;
    Eigen::Matrix<double, 6, 6> cov;
    double timestamp;

    if (ekf_->getFusedPose(pose, cov, timestamp)) {
        rclcpp::Time stamp(timestamp, RCL_ROS_TIME);

        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.stamp = stamp;
        odom_msg.header.frame_id = odom_frame_;
        odom_msg.child_frame_id = base_frame_;

        odom_msg.pose.pose.position.x = pose(0, 3);
        odom_msg.pose.pose.position.y = pose(1, 3);
        odom_msg.pose.pose.position.z = pose(2, 3);

        Eigen::Quaterniond q(pose.block<3,3>(0,0));
        odom_msg.pose.pose.orientation.w = q.w();
        odom_msg.pose.pose.orientation.x = q.x();
        odom_msg.pose.pose.orientation.y = q.y();
        odom_msg.pose.pose.orientation.z = q.z();

        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                odom_msg.pose.covariance[i * 6 + j] = cov(i, j);
            }
        }

        fused_odom_pub_->publish(odom_msg);

        geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
        pose_msg.header.stamp = stamp;
        pose_msg.header.frame_id = odom_frame_;
        pose_msg.pose = odom_msg.pose;
        fused_pose_pub_->publish(pose_msg);

        if (publish_tf_) {
            geometry_msgs::msg::TransformStamped tf_msg;
            tf_msg.header.stamp = stamp;
            tf_msg.header.frame_id = odom_frame_;
            tf_msg.child_frame_id = base_frame_;
            tf_msg.transform.translation.x = pose(0, 3);
            tf_msg.transform.translation.y = pose(1, 3);
            tf_msg.transform.translation.z = pose(2, 3);
            tf_msg.transform.rotation.w = q.w();
            tf_msg.transform.rotation.x = q.x();
            tf_msg.transform.rotation.y = q.y();
            tf_msg.transform.rotation.z = q.z();
            tf_broadcaster_->sendTransform(tf_msg);
        }
    }
}

}  // namespace lunar_slam

RCLCPP_COMPONENTS_REGISTER_NODE(lunar_slam::SensorFusionNode)