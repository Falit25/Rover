#include "lunar_slam/lidar_odometry_node.hpp"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

namespace lunar_slam {

LidarOdometryNode::LidarOdometryNode(const rclcpp::NodeOptions& options)
    : Node("lidar_odometry_node", options) {

    this->declare_parameter("cloud_topic", "/velodyne_points");
    this->declare_parameter("odom_topic", "/lidar_odom");
    this->declare_parameter("base_frame", "base_link");
    this->declare_parameter("odom_frame", "odom");
    this->declare_parameter("lidar_frame", "lidar_link");
    this->declare_parameter("publish_tf", false);
    this->declare_parameter("voxel_leaf_size", 0.1);
    this->declare_parameter("max_correspondence_distance", 1.0);
    this->declare_parameter("max_iterations", 30);

    config_.voxel_leaf_size = this->get_parameter("voxel_leaf_size").as_double();
    config_.max_correspondence_distance = this->get_parameter("max_correspondence_distance").as_double();
    config_.max_iterations = this->get_parameter("max_iterations").as_int();

std::string cloud_topic = this->get_parameter("cloud_topic").as_string();
    std::string odom_topic = this->get_parameter("odom_topic").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();
    odom_frame_ = this->get_parameter("odom_frame").as_string();
    lidar_frame_ = this->get_parameter("lidar_frame").as_string();
    publish_tf_ = this->get_parameter("publish_tf").as_bool();

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic, 10, std::bind(&LidarOdometryNode::cloudCallback, this, std::placeholders::_1));

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(odom_topic, 10);

    if (publish_tf_) {
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    prev_cloud_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

    RCLCPP_INFO(this->get_logger(), "LiDAR Odometry Node initialized");
}

void LidarOdometryNode::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->empty()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Empty point cloud received");
        return;
    }

    // Downsample
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(config_.voxel_leaf_size, config_.voxel_leaf_size, config_.voxel_leaf_size);
    voxel_grid.filter(*filtered);

    // Remove outliers
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(filtered);
    sor.setMeanK(20);
    sor.setStddevMulThresh(1.0);
    sor.filter(*filtered);

    if (filtered->size() < 50) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Too few points after filtering");
        return;
    }

    Eigen::Matrix4d pose = last_pose_;

    if (initialized_) {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(filtered);
        icp.setInputTarget(prev_cloud_);
        icp.setMaxCorrespondenceDistance(config_.max_correspondence_distance);
        icp.setMaximumIterations(config_.max_iterations);
        icp.setTransformationEpsilon(config_.transformation_epsilon);
        icp.setEuclideanFitnessEpsilon(config_.euclidean_fitness_epsilon);
        icp.setInitialAlignment(Eigen::Matrix4f::Identity());

        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
        icp.align(*aligned);

        if (icp.hasConverged() && icp.getFitnessScore() < 1.0) {
            Eigen::Matrix4d delta = icp.getFinalTransformation().cast<double>();
            // delta is transformation from source (current) to target (previous)
            // We need the forward motion in LiDAR frame: previous -> current = delta.inverse()
            Eigen::Matrix4d motion_lidar = delta.inverse();
            
            // Transform LiDAR motion to base_link frame using static extrinsics
            // T_base = T_lidar_to_base * T_lidar * T_lidar_to_base.inverse()
            Eigen::Matrix4d motion_base;
            try {
                geometry_msgs::msg::TransformStamped lidar_to_base = tf_buffer_->lookupTransform(
                    base_frame_, lidar_frame_, tf2::TimePointZero);
                Eigen::Matrix4d T_lidar_to_base = tf2::transformToEigen(lidar_to_base.transform).matrix();
                Eigen::Matrix4d motion_base_unscaled = T_lidar_to_base * motion_lidar * T_lidar_to_base.inverse();
                motion_base = motion_base_unscaled;
            } catch (tf2::TransformException& ex) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "TF lookup failed (lidar->base): %s, using LiDAR frame motion", ex.what());
                motion_base = motion_lidar;
            }
            
            pose = last_pose_ * motion_base;
            
            // Check for degenerate motion (pure rotation)
            Eigen::Vector3d trans = motion_base.block<3,1>(0,3);
            if (trans.norm() < 0.01 && motion_base.block<3,3>(0,0).isIdentity(1e-3)) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Degenerate ICP motion detected");
            }
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "ICP did not converge, fitness: %.3f", icp.getFitnessScore());
        }
    } else {
        initialized_ = true;
    }

    last_pose_ = pose;
    prev_cloud_ = filtered;
    publishOdom(pose, msg->header.stamp);
}

void LidarOdometryNode::publishOdom(const Eigen::Matrix4d& pose, const rclcpp::Time& stamp) {
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

    // Set covariance (low for LiDAR odometry)
    for (int i = 0; i < 6; ++i) {
        odom_msg.pose.covariance[i * 6 + i] = 0.01;
    }

    odom_pub_->publish(odom_msg);

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

}  // namespace lunar_slam

RCLCPP_COMPONENTS_REGISTER_NODE(lunar_slam::LidarOdometryNode)