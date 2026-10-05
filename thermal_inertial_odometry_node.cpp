#include "lunar_slam/thermal_inertial_odometry_node.hpp"

namespace lunar_slam {

ThermalInertialOdometryNode::ThermalInertialOdometryNode(const rclcpp::NodeOptions& options)
    : Node("thermal_inertial_odometry_node", options) {

    this->declare_parameter("imu_topic", "/imu/data");
    this->declare_parameter("thermal_topic", "/thermal/image_raw");
    this->declare_parameter("odom_topic", "/thermal_inertial_odom");
    this->declare_parameter("pose_topic", "/thermal_inertial_pose");
    this->declare_parameter("base_frame", "base_link");
    this->declare_parameter("odom_frame", "odom");
    this->declare_parameter("publish_tf", false);
    this->declare_parameter("timer_rate", 20.0);

    config_.img_rate = this->declare_parameter("thermal_inertial.img_rate", 10.0);
    config_.imu_rate = this->declare_parameter("thermal_inertial.imu_rate", 200.0);
    config_.max_features = this->declare_parameter("thermal_inertial.max_features", 500);
    config_.feature_min_distance = this->declare_parameter("thermal_inertial.feature_min_distance", 15.0);
    config_.ransac_threshold = this->declare_parameter("thermal_inertial.ransac_threshold", 2.0);
    config_.parallax_threshold = this->declare_parameter("thermal_inertial.parallax_threshold", 10.0);
    config_.gravity = Eigen::Vector3d(
        this->declare_parameter("thermal_inertial.gravity_x", 0.0),
        this->declare_parameter("thermal_inertial.gravity_y", 0.0),
        this->declare_parameter("thermal_inertial.gravity_z", -1.62)
    );
    config_.accel_noise = this->declare_parameter("thermal_inertial.accel_noise", 1e-3);
    config_.gyro_noise = this->declare_parameter("thermal_inertial.gyro_noise", 1e-4);
    config_.accel_bias_noise = this->declare_parameter("thermal_inertial.accel_bias_noise", 1e-5);
    config_.gyro_bias_noise = this->declare_parameter("thermal_inertial.gyro_bias_noise", 1e-6);

    tio_ = std::make_unique<ThermalInertialOdometry>(config_);

    std::string imu_topic = this->get_parameter("imu_topic").as_string();
    std::string thermal_topic = this->get_parameter("thermal_topic").as_string();
    std::string odom_topic = this->get_parameter("odom_topic").as_string();
    std::string pose_topic = this->get_parameter("pose_topic").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();
    odom_frame_ = this->get_parameter("odom_frame").as_string();
    publish_tf_ = this->get_parameter("publish_tf").as_bool();
    double timer_rate = this->get_parameter("timer_rate").as_double();

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic, 100, std::bind(&ThermalInertialOdometryNode::imuCallback, this, std::placeholders::_1));

    thermal_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        thermal_topic, 10, std::bind(&ThermalInertialOdometryNode::thermalCallback, this, std::placeholders::_1));

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(odom_topic, 10);
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(pose_topic, 10);

    if (publish_tf_) {
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / timer_rate),
        std::bind(&ThermalInertialOdometryNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "Thermal-Inertial Odometry Node initialized");
}

void ThermalInertialOdometryNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    tio_->processIMU(msg);
}

void ThermalInertialOdometryNode::thermalCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    tio_->processThermalImage(msg);
}

void ThermalInertialOdometryNode::timerCallback() {
    Eigen::Matrix4d pose;
    double timestamp;
    
    if (tio_->getPose(pose, timestamp)) {
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

        odom_pub_->publish(odom_msg);

        geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
        pose_msg.header.stamp = stamp;
        pose_msg.header.frame_id = odom_frame_;
        pose_msg.pose.pose = odom_msg.pose.pose;
        pose_pub_->publish(pose_msg);

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

RCLCPP_COMPONENTS_REGISTER_NODE(lunar_slam::ThermalInertialOdometryNode)