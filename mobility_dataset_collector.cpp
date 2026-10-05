#include "lunar_slam/mobility_dataset_collector.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace lunar_slam {

MobilityDatasetCollector::MobilityDatasetCollector(const rclcpp::NodeOptions& options)
    : Node("mobility_dataset_collector", options) {

    this->declare_parameter("output_file", "mobility_dataset.csv");
    this->declare_parameter("slip_threshold", 0.15);
    this->declare_parameter("sinkage_threshold", 0.05);
    this->declare_parameter("record_rate", 2.0);
    this->declare_parameter("buffer_size", 100);
    this->declare_parameter("record_thermal", true);
    this->declare_parameter("robot_base_frame", "base_link");
    this->declare_parameter("odom_frame", "odom");
    this->declare_parameter("gazebo_ground_truth_topic", "/gazebo/ground_truth/pose");
    this->declare_parameter("wheel_joint_names", std::vector<std::string>{"rear_left_wheel_joint", "rear_right_wheel_joint"});
    this->declare_parameter("wheel_radius", 0.15);

    config_.output_file = this->get_parameter("output_file").as_string();
    config_.slip_threshold = this->get_parameter("slip_threshold").as_double();
    config_.sinkage_threshold = this->get_parameter("sinkage_threshold").as_double();
    config_.record_rate = this->get_parameter("record_rate").as_double();
    config_.buffer_size = this->get_parameter("buffer_size").as_int();
    config_.record_thermal = this->get_parameter("record_thermal").as_bool();
    config_.robot_base_frame = this->get_parameter("robot_base_frame").as_string();
    config_.odom_frame = this->get_parameter("odom_frame").as_string();
    config_.gazebo_ground_truth_topic = this->get_parameter("gazebo_ground_truth_topic").as_string();
    config_.elevation_map_topic = this->get_parameter("elevation_map_topic").as_string();
    config_.wheel_joint_names = this->get_parameter("wheel_joint_names").as_string_array();
    config_.wheel_radius = this->get_parameter("wheel_radius").as_double();
    config_.elevation_map_resolution = this->get_parameter("elevation_map_resolution").as_double();

    // Open CSV file
    csv_file_.open(config_.output_file);
    if (!csv_file_.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open output file: %s", config_.output_file.c_str());
    } else {
        // Write header
        csv_file_ << "timestamp,elevation,slope_deg,curvature,roughness,step_height,"
                  << "temperature_k,thermal_contrast,thermal_gradient,"
                  << "velocity,angular_velocity,wheel_torque,"
                  << "slip_probability,traction_loss,immobilization_probability\n";
        RCLCPP_INFO(this->get_logger(), "Recording dataset to: %s", config_.output_file.c_str());
    }

    // Subscribers
    auto qos = rclcpp::SensorDataQoS();
    
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/velodyne_points", qos, 
        std::bind(&MobilityDatasetCollector::cloudCallback, this, std::placeholders::_1));
    
    thermal_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/thermal/image_raw", qos,
        std::bind(&MobilityDatasetCollector::thermalCallback, this, std::placeholders::_1));
    
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data", qos,
        std::bind(&MobilityDatasetCollector::imuCallback, this, std::placeholders::_1));
    
    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 10,
        std::bind(&MobilityDatasetCollector::jointStateCallback, this, std::placeholders::_1));
    
    ground_truth_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        config_.gazebo_ground_truth_topic, 10,
        std::bind(&MobilityDatasetCollector::groundTruthCallback, this, std::placeholders::_1));
    
elevation_map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        config_.elevation_map_topic, 10,
        std::bind(&MobilityDatasetCollector::elevationMapCallback, this, std::placeholders::_1));
    
    // TF
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Timer for periodic recording
    timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / config_.record_rate),
        std::bind(&MobilityDatasetCollector::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "Mobility Dataset Collector initialized with Gazebo ground truth");
}

void MobilityDatasetCollector::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    latest_cloud_ = msg;
}

void MobilityDatasetCollector::thermalCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    try {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "mono16");
        latest_thermal_ = cv_ptr->image.clone();
    } catch (cv_bridge::Exception& e) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Thermal conversion failed: %s", e.what());
    }
}

void MobilityDatasetCollector::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    latest_imu_ = msg;
}

void MobilityDatasetCollector::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    prev_joint_state_ = latest_joint_state_;
    latest_joint_state_ = msg;
}

void MobilityDatasetCollector::groundTruthCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    prev_ground_truth_ = latest_ground_truth_;
    latest_ground_truth_ = msg;
}

void MobilityDatasetCollector::wheelOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    prev_wheel_odom_ = latest_wheel_odom_;
    latest_wheel_odom_ = msg;
}

void MobilityDatasetCollector::fusedOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    prev_fused_odom_ = latest_fused_odom_;
    latest_fused_odom_ = msg;
}

void MobilityDatasetCollector::cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    latest_cmd_vel_ = msg;
}

void MobilityDatasetCollector::costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    latest_costmap_ = msg;
}

void MobilityDatasetCollector::elevationMapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    latest_elevation_map_ = msg;
}

void MobilityDatasetCollector::timerCallback() {
    processAndRecord();
    
    // Flush buffer periodically
    if (write_buffer_.size() >= config_.buffer_size) {
        flushBuffer();
    }
}

void MobilityDatasetCollector::processAndRecord() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    if (!latest_cloud_ || !latest_ground_truth_ || !latest_joint_state_ || !latest_costmap_) {
        return;
    }
    
    // Need previous states to compute slip
    if (!prev_joint_state_ || !prev_ground_truth_) {
        prev_joint_state_ = latest_joint_state_;
        prev_ground_truth_ = latest_ground_truth_;
        prev_time_ = this->now();
        return;
    }
    
    // Transform point cloud from lidar_link to odom frame
    sensor_msgs::msg::PointCloud2 cloud_odom;
    try {
        cloud_odom = tf_buffer_->transform(*latest_cloud_, odom_frame_, tf2::durationFromSec(0.1));
    } catch (tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
            "TF transform failed (lidar->odom): %s", ex.what());
        return;
    }
    
    // Compute REAL slip from Gazebo joint states and ground truth
    double slip = computeWheelSlipFromGazebo();
    double sinkage = computeSinkage();
    
    // Extract terrain features from LiDAR (real elevation map) - use transformed cloud
    TerrainFeatures features = extractFeaturesFromLidar(cloud_odom);
    
    // Create ground truth labels
    MobilityRiskPrediction gt = createGroundTruth(slip, sinkage);
    
    // Write to buffer
    write_buffer_.emplace_back(features, gt);
    
    // Also log to CSV immediately for safety
    writeToCSV(features, gt);
    
    // Update previous states
    prev_joint_state_ = latest_joint_state_;
    prev_ground_truth_ = latest_ground_truth_;
    prev_time_ = this->now();
}

double MobilityDatasetCollector::computeWheelSlipFromGazebo() const {
    if (!prev_joint_state_ || !latest_joint_state_ || !prev_ground_truth_ || !latest_ground_truth_) {
        return 0.0;
    }
    
    // Time delta
    double dt = (this->now() - prev_time_).seconds();
    if (dt <= 0 || dt > 1.0) return 0.0;
    
    // Find wheel joint indices
    int left_idx = -1, right_idx = -1;
    for (size_t i = 0; i < latest_joint_state_->name.size(); ++i) {
        if (latest_joint_state_->name[i] == config_.wheel_joint_names[0]) left_idx = i;
        if (latest_joint_state_->name[i] == config_.wheel_joint_names[1]) right_idx = i;
    }
    if (left_idx == -1 || right_idx == -1) return 0.0;
    
    // Wheel angular velocities (rad/s)
    double left_omega = latest_joint_state_->velocity[left_idx];
    double right_omega = latest_joint_state_->velocity[right_idx];
    
    // Previous wheel angles
    double left_omega_prev = prev_joint_state_->velocity[left_idx];
    double right_omega_prev = prev_joint_state_->velocity[right_idx];
    
    // Average wheel speed (m/s)
    double wheel_speed = (std::abs(left_omega) + std::abs(right_omega)) / 2.0 * config_.wheel_radius;
    
    // Body velocity from Gazebo ground truth (use twist for velocity)
    double body_speed = std::sqrt(
        latest_ground_truth_->twist.twist.linear.x * latest_ground_truth_->twist.twist.linear.x +
        latest_ground_truth_->twist.twist.linear.y * latest_ground_truth_->twist.twist.linear.y);
    
    // Slip ratio = (wheel_speed - body_speed) / max(wheel_speed, epsilon)
    if (wheel_speed > 0.01) {
        double slip_ratio = (wheel_speed - body_speed) / wheel_speed;
        return std::clamp(slip_ratio, -1.0, 1.0);
    }
    
    return 0.0;
}

double MobilityDatasetCollector::computeSinkage() const
{
    if (!latest_ground_truth_ || !latest_costmap_) {
        return 0.0;
    }

    const double wheel_radius = 0.15;

    const std::array<Eigen::Vector3d, 4> wheel_offsets = {
        Eigen::Vector3d( 0.30,  0.22, 0.0), // FL
        Eigen::Vector3d( 0.30, -0.22, 0.0), // FR
        Eigen::Vector3d(-0.30,  0.22, 0.0), // RL
        Eigen::Vector3d(-0.30, -0.22, 0.0)  // RR
    };

    double total_sinkage = 0.0;
    int valid_wheels = 0;

    const auto& pose = latest_ground_truth_->pose.pose;

    tf2::Quaternion q(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w);

    tf2::Matrix3x3 rotation(q);

    for (const auto& offset : wheel_offsets) {

        // Transform wheel offset from base_link → world
        tf2::Vector3 local_offset(
            offset.x(),
            offset.y(),
            offset.z());

        tf2::Vector3 world_offset =
            tf2::quatRotate(q, local_offset);

        double wx =
            pose.position.x +
            world_offset.x();

        double wy =
            pose.position.y +
            world_offset.y();

        double wheel_center_z =
            pose.position.z +
            world_offset.z();

        // Terrain directly underneath this wheel
        double terrain_z =
            getTerrainHeight(wx, wy);

        if (!std::isfinite(terrain_z)) {
            continue;
        }

        // Wheel center if wheel is just touching terrain
        double expected_center_z =
            terrain_z + wheel_radius;

        double sinkage =
            expected_center_z - wheel_center_z;

        // Negative means wheel is above terrain,
        // not sinkage.
        sinkage = std::max(0.0, sinkage);

        // Prevent unrealistic values
        sinkage = std::min(0.20, sinkage);

        total_sinkage += sinkage;
        valid_wheels++;
    }

    if (valid_wheels == 0) {
        return 0.0;
    }

    return total_sinkage /
           static_cast<double>(valid_wheels);
}

double MobilityDatasetCollector::getTerrainHeight(double x, double y) const {
    if (!latest_elevation_map_) return std::numeric_limits<double>::quiet_NaN();
    
    const auto& map = *latest_elevation_map_;
    double res = map.info.resolution;
    double ox = map.info.origin.position.x;
    double oy = map.info.origin.position.y;
    
    int gx = static_cast<int>((x - ox) / res);
    int gy = static_cast<int>((y - oy) / res);
    
    if (gx < 0 || gy < 0 || gx >= (int)map.info.width || gy >= (int)map.info.height) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    int idx = gy * map.info.width + gx;
    int16_t cell = map.data[idx];
    
    if (cell < 0) return std::numeric_limits<double>::quiet_NaN();
    
    // Elevation map stores height in centimeters (int16)
    return map.info.origin.position.z + (cell / 100.0);
}

TerrainFeatures MobilityDatasetCollector::extractFeaturesFromLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) const {
    TerrainFeatures features;
    
    if (!cloud_msg) return features;
    
    // Get robot pose from ground truth (Odometry message)
    double rx = latest_ground_truth_->pose.pose.position.x;
    double ry = latest_ground_truth_->pose.pose.position.y;
    double rz = latest_ground_truth_->pose.pose.position.z;
    
    // Velocity from ground truth twist
    features.velocity = std::sqrt(
        latest_ground_truth_->twist.twist.linear.x * latest_ground_truth_->twist.twist.linear.x +
        latest_ground_truth_->twist.twist.linear.y * latest_ground_truth_->twist.twist.linear.y);
    features.angular_velocity = latest_ground_truth_->twist.twist.angular.z;
    features.wheel_torque = 0.0;
    features.sinkage = computeSinkage();
    
    // Convert point cloud to elevation map
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    
    if (cloud->empty()) return features;
    
    // Create local elevation grid around robot
    const double local_res = 0.1;
    const int local_size = 50; // 5m x 5m
    const double local_origin_x = rx - local_size * local_res / 2.0;
    const double local_origin_y = ry - local_size * local_res / 2.0;
    
    std::vector<double> elevations(local_size * local_size, -1e9);
    std::vector<int> counts(local_size * local_size, 0);
    
    for (const auto& point : cloud->points) {
        int gx = static_cast<int>((point.x - local_origin_x) / 0.1);
        int gy = static_cast<int>((point.y - local_origin_y) / 0.1);
        if (gx >= 0 && gx < local_size && gy >= 0 && gy < local_size) {
            int idx = gy * local_size + gx;
            elevations[idx] = std::max(elevations[idx], static_cast<double>(point.z));
            counts[idx]++;
        }
    }
    
    // Find robot cell
    int robot_gx = static_cast<int>((rx - local_origin_x) / local_res);
    int robot_gy = static_cast<int>((ry - local_origin_y) / local_res);
    
    if (robot_gx >= 2 && robot_gx < local_size - 2 && robot_gy >= 2 && robot_gy < local_size - 2) {
        int robot_idx = robot_gy * local_size + robot_gx;
        
        // Elevation at robot position
        if (counts[robot_idx] > 0) {
            features.elevation = elevations[robot_idx];
        }
        
        // Compute slope, roughness, step_height in 3x3 neighborhood
        std::vector<double> neighbor_elevations;
        double max_elev = -1e9, min_elev = 1e9;
        double dz_dx = 0.0, dz_dy = 0.0;
        int valid = 0;
        
        for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
                if (kx == 0 && ky == 0) continue;
                int nx = robot_gx + kx;
                int ny = robot_gy + ky;
                int nidx = ny * local_size + nx;
                if (counts[nidx] > 0) {
                    double elev = elevations[nidx];
                    neighbor_elevations.push_back(elev);
                    max_elev = std::max(max_elev, elev);
                    min_elev = std::min(min_elev, elev);
                    
                    double dz = elev - elevations[robot_idx];
                    dz_dx += dz * kx;
                    dz_dy += dz * ky;
                    valid++;
                }
            }
            
            if (valid > 0) {
                double grad_mag = std::sqrt(dz_dx * dz_dx + dz_dy * dz_dy) / (local_res * valid);
                features.slope_deg = std::atan(grad_mag) * 180.0 / M_PI;
            }
            
            if (!neighbor_elevations.empty()) {
                double mean = 0.0;
                for (double e : neighbor_elevations) mean += e;
                mean /= neighbor_elevations.size();
                double var = 0.0;
                for (double e : neighbor_elevations) var += (e - mean) * (e - mean);
                features.roughness = std::sqrt(var / neighbor_elevations.size());
            }
            
            if (max_elev > -1e8 && min_elev < 1e8) {
                features.step_height = max_elev - min_elev;
            }
        }
    }
    
    // Thermal features
    if (config_.record_thermal && !latest_thermal_.empty()) {
        int cx = latest_thermal_.cols / 2;
        int cy = latest_thermal_.rows / 2;
        if (cx >= 0 && cx < latest_thermal_.cols && cy >= 0 && cy < latest_thermal_.rows) {
            features.temperature_k = latest_thermal_.at<uint16_t>(cy, cx) / 100.0;
            
            // Thermal contrast in 11x11 region
            double sum = 0, sum_sq = 0, count = 0;
            for (int dy = -5; dy <= 5; ++dy) {
                for (int dx = -5; dx <= 5; ++dx) {
                    int x = cx + dx, y = cy + dy;
                    if (x >= 0 && x < latest_thermal_.cols && y >= 0 && y < latest_thermal_.rows) {
                        double t = latest_thermal_.at<uint16_t>(y, x) / 100.0;
                        sum += t;
                        sum_sq += t * t;
                        count++;
                    }
                }
            }
            if (count > 0) {
                double mean = sum / count;
                double var = sum_sq / count - mean * mean;
                features.thermal_contrast = std::sqrt(var);
            }
        }
    }
    
    // Rover state
    if (latest_ground_truth_) {
        // Velocity would need previous ground truth - simplified for now
        features.velocity = 0.0;
    }
    if (latest_cmd_vel_) {
        features.wheel_torque = std::abs(latest_cmd_vel_->linear.x) * 10.0;
    }
    
    return features;
}

MobilityRiskPrediction MobilityDatasetCollector::createGroundTruth(double slip, double sinkage) const {
    MobilityRiskPrediction gt;
    gt.slip_probability = std::min(1.0, std::abs(slip) / config_.slip_threshold);
    gt.sinkage_risk = std::min(1.0, sinkage / config_.sinkage_threshold);
    gt.traction_loss = gt.slip_probability * 0.8;
    gt.immobilization_probability = std::min(1.0, (gt.slip_probability + gt.sinkage_risk) / 2.0);
    gt.confidence = 1.0;
    gt.prediction_uncertainty = 0.0;
    return gt;
}

void MobilityDatasetCollector::writeToCSV(const TerrainFeatures& features, const MobilityRiskPrediction& gt) {
    if (!csv_file_.is_open()) return;
    
    auto vec = features.toVector();
    csv_file_ << this->now().seconds() << ",";
    for (size_t i = 0; i < vec.size(); ++i) {
        csv_file_ << vec[i] << ",";
    }
    csv_file_ << gt.slip_probability << ","
              << gt.sinkage_risk << ","
              << gt.traction_loss << ","
              << gt.immobilization_probability << "\n";
    csv_file_.flush();
}

void MobilityDatasetCollector::flushBuffer() {
    write_buffer_.clear();
}

TerrainFeatures MobilityDatasetCollector::extractFeaturesAtRobot() const {
    // Deprecated - use extractFeaturesFromLidar()
    return TerrainFeatures();
}

}  // namespace lunar_slam

RCLCPP_COMPONENTS_REGISTER_NODE(lunar_slam::MobilityDatasetCollector)