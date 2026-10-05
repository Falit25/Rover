#include "lunar_slam/hazard_detector_node.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace lunar_slam {

HazardDetectorNode::HazardDetectorNode(const rclcpp::NodeOptions& options)
    : Node("hazard_detector_node", options) {

    this->declare_parameter("lidar_topic", "/velodyne_points");
    this->declare_parameter("thermal_topic", "/thermal/image_raw");
    this->declare_parameter("thermal_camera_info_topic", "/thermal/camera_info");
    this->declare_parameter("hazard_map_topic", "/hazard_map");
    this->declare_parameter("costmap_topic", "/traversability_costmap");
    this->declare_parameter("slope_map_topic", "/slope_map");
    this->declare_parameter("slip_map_topic", "/slip_map");
    this->declare_parameter("timer_rate", 5.0);
    this->declare_parameter("lidar_frame", "lidar_link");
    this->declare_parameter("thermal_frame", "thermal_camera_link");
    this->declare_parameter("base_frame", "base_link");
    this->declare_parameter("odom_frame", "odom");
    this->declare_parameter("use_mobility_risk", true);
    this->declare_parameter("mobility_risk_weight", 0.7);
    this->declare_parameter("hazard_weight", 0.3);
    this->declare_parameter("uncertainty_weight", 0.2);
    this->declare_parameter("mobility_risk_model_path", "");  // NEW: weight for uncertainty penalty

    config_.resolution = this->declare_parameter("hazard_detector.resolution", 0.1);
    config_.grid_size = this->declare_parameter("hazard_detector.grid_size", 200);
    config_.max_step_height = this->declare_parameter("hazard_detector.max_step_height", 0.15);
    config_.max_slope_deg = this->declare_parameter("hazard_detector.max_slope_deg", 20.0);
    config_.crater_rim_curvature = this->declare_parameter("hazard_detector.crater_rim_curvature", 0.5);
    config_.rock_height_thresh = this->declare_parameter("hazard_detector.rock_height_thresh", 0.2);
    config_.cold_trap_temp_thresh = this->declare_parameter("hazard_detector.cold_trap_temp_thresh", 100.0);
    config_.thermal_noise_sigma = this->declare_parameter("hazard_detector.thermal_noise_sigma", 5.0);

    hazard_detector_ = std::make_unique<HazardDetector>(config_);

    // Initialize mobility risk model
    MobilityRiskModel::Config risk_config;
    risk_config.online_learning = true;
    risk_config.n_estimators = 50;
    risk_config.max_depth = 8;
    mobility_risk_model_ = std::make_unique<MobilityRiskModel>(risk_config);

    // Load trained mobility model if path provided
    std::string model_path = this->get_parameter("mobility_risk_model_path").as_string();
    if (!model_path.empty()) {
        if (!mobility_risk_model_->load(model_path)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load mobility model: %s", model_path.c_str());
        } else {
            RCLCPP_INFO(this->get_logger(), "Mobility model loaded from: %s", model_path.c_str());
        }
    }

    std::string lidar_topic = this->get_parameter("lidar_topic").as_string();
    std::string thermal_topic = this->get_parameter("thermal_topic").as_string();
    std::string thermal_camera_info_topic = this->get_parameter("thermal_camera_info_topic").as_string();
    std::string hazard_map_topic = this->get_parameter("hazard_map_topic").as_string();
    std::string costmap_topic = this->get_parameter("costmap_topic").as_string();
    std::string slope_map_topic = this->get_parameter("slope_map_topic").as_string();
    std::string surface_risk_map_topic = this->get_parameter("surface_risk_map_topic").as_string();
    double timer_rate = this->get_parameter("timer_rate").as_double();
    lidar_frame_ = this->get_parameter("lidar_frame").as_string();
    thermal_frame_ = this->get_parameter("thermal_frame").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();
    odom_frame_ = this->get_parameter("odom_frame").as_string();
    use_mobility_risk_ = this->get_parameter("use_mobility_risk").as_bool();
    mobility_risk_weight_ = this->get_parameter("mobility_risk_weight").as_double();
    hazard_weight_ = this->get_parameter("hazard_weight").as_double();
    uncertainty_weight_ = this->get_parameter("uncertainty_weight").as_double();

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_topic, 10, std::bind(&HazardDetectorNode::lidarCallback, this, std::placeholders::_1));

    thermal_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        thermal_topic, 10, std::bind(&HazardDetectorNode::thermalCallback, this, std::placeholders::_1));

    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        thermal_camera_info_topic, 10, std::bind(&HazardDetectorNode::cameraInfoCallback, this, std::placeholders::_1));

    // Subscribe to fused odometry for real rover state
    fused_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/fused_odom", 10, std::bind(&HazardDetectorNode::fusedOdomCallback, this, std::placeholders::_1));

    hazard_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(hazard_map_topic, 10);
    costmap_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(costmap_topic, 10);
    slope_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(slope_map_topic, 10);
    surface_risk_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(surface_risk_map_topic, 10);

    timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / timer_rate),
        std::bind(&HazardDetectorNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "Hazard Detector Node initialized with Mobility Risk Model");
}

void HazardDetectorNode::fusedOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(fused_odom_mutex_);
    latest_fused_odom_ = msg;
}

void HazardDetectorNode::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    cv::Mat thermal;
    sensor_msgs::msg::CameraInfo camera_info;
    bool have_camera_info = false;
    {
        std::lock_guard<std::mutex> lock(thermal_mutex_);
        if (!latest_thermal_.empty()) {
            thermal = latest_thermal_.clone();
        }
        if (have_camera_info_) {
            camera_info = latest_camera_info_;
            have_camera_info = true;
        }
    }

    // Get transform from odom frame to thermal camera frame (since cloud is in odom frame)
    geometry_msgs::msg::TransformStamped odom_to_thermal;
    try {
        odom_to_thermal = tf_buffer_->lookupTransform(thermal_frame_, odom_frame_, tf2::TimePointZero);
    } catch (tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
            "TF lookup failed (odom->thermal): %s", ex.what());
        return;
    }

    // Get robot pose in odom frame for mobility risk
    geometry_msgs::msg::TransformStamped odom_to_base;
    try {
        odom_to_base = tf_buffer_->lookupTransform(odom_frame_, base_frame_, tf2::TimePointZero);
    } catch (tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
            "TF lookup failed (odom->base): %s", ex.what());
    }

    // Transform point cloud from lidar_link frame to odom frame
    sensor_msgs::msg::PointCloud2 cloud_odom;
    try {
        cloud_odom = tf_buffer_->transform(*msg, odom_frame_, tf2::durationFromSec(0.1));
    } catch (tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
            "TF transform failed (lidar->odom): %s", ex.what());
        return;
    }

    auto hazard_grid = hazard_detector_->detectHazards(cloud_odom, thermal.empty() ? nullptr : &thermal, 
                                                        have_camera_info ? &camera_info : nullptr, &odom_to_thermal);
    
    // Generate slope and surface risk maps from transformed cloud (in odom frame)
    auto cloud_odom_ptr = std::make_shared<sensor_msgs::msg::PointCloud2>(cloud_odom);
    auto slope_grid = generateSlopeMap(*cloud_odom_ptr);
    auto surface_risk_grid = generateSurfaceRiskMap(*cloud_odom_ptr);
    
    // Enhance with mobility risk if enabled
    if (use_mobility_risk_ && mobility_risk_model_ && mobility_risk_model_->isTrained()) {
        enhanceCostmapWithMobilityRisk(hazard_grid, odom_to_base);
    }
    
    hazard_map_pub_->publish(hazard_grid);
    costmap_pub_->publish(hazard_grid);
    slope_map_pub_->publish(slope_grid);
    surface_risk_map_pub_->publish(surface_risk_grid);
}

nav_msgs::msg::OccupancyGrid HazardDetectorNode::generateSlopeMap(
    const sensor_msgs::msg::PointCloud2& cloud_msg) {
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(cloud_msg, *cloud);

    nav_msgs::msg::OccupancyGrid grid_msg;
    grid_msg.header = cloud_msg->header;
    grid_msg.header.frame_id = "odom";
    grid_msg.info.resolution = config_.resolution;
    grid_msg.info.width = config_.grid_size;
    grid_msg.info.height = config_.grid_size;
    grid_msg.info.origin.position.x = -config_.grid_size * config_.resolution / 2.0;
    grid_msg.info.origin.position.y = -config_.grid_size * config_.resolution / 2.0;
    grid_msg.info.origin.position.z = 0.0;
    grid_msg.info.origin.orientation.w = 1.0;

    grid_msg.data.resize(config_.grid_size * config_.grid_size, -1);

    double origin_offset = config_.grid_size * config_.resolution / 2.0;
    std::vector<double> elevations(config_.grid_size * config_.grid_size, -1e9);
    std::vector<int> counts(config_.grid_size * config_.grid_size, 0);

    for (const auto& point : cloud->points) {
        int gx = static_cast<int>((point.x + origin_offset) / config_.resolution);
        int gy = static_cast<int>((point.y + origin_offset) / config_.resolution);
        if (gx >= 0 && gx < config_.grid_size && gy >= 0 && gy < config_.grid_size) {
            int idx = gy * config_.grid_size + gx;
            elevations[idx] = std::max(elevations[idx], static_cast<double>(point.z));
            counts[idx]++;
        }
    }

    // Compute slopes
    for (int y = 1; y < config_.grid_size - 1; ++y) {
        for (int x = 1; x < config_.grid_size - 1; ++x) {
            int idx = y * config_.grid_size + x;
            if (counts[idx] == 0) continue;

            double dz_dx = 0.0, dz_dy = 0.0;
            int valid = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    if (kx == 0 && ky == 0) continue;
                    int nx = x + kx, ny = y + ky;
                    int nidx = ny * config_.grid_size + nx;
                    if (counts[nidx] > 0) {
                        double dz = elevations[nidx] - elevations[idx];
                        dz_dx += dz * kx;
                        dz_dy += dz * ky;
                        valid++;
                    }
                }
            }
            if (valid > 0) {
                double slope = std::atan(std::sqrt(dz_dx*dz_dx + dz_dy*dz_dy) / (config_.resolution * valid)) * 180.0 / M_PI;
                int slope_cost = std::min(100, static_cast<int>(slope / 30.0 * 100)); // 30 deg = 100
                grid_msg.data[idx] = slope_cost;
            }
        }
    }

    return grid_msg;
}

nav_msgs::msg::OccupancyGrid HazardDetectorNode::generateSurfaceRiskMap(
    const sensor_msgs::msg::PointCloud2& cloud_msg) {
    
    nav_msgs::msg::OccupancyGrid grid_msg;
    grid_msg.header = cloud_msg.header;
    grid_msg.header.frame_id = "odom";
    grid_msg.info.resolution = config_.resolution;
    grid_msg.info.width = config_.grid_size;
    grid_msg.info.height = config_.grid_size;
    grid_msg.info.origin.position.x = -config_.grid_size * config_.resolution / 2.0;
    grid_msg.info.origin.position.y = -config_.grid_size * config_.resolution / 2.0;
    grid_msg.info.origin.position.z = 0.0;
    grid_msg.info.origin.orientation.w = 1.0;

    grid_msg.data.resize(config_.grid_size * config_.grid_size, -1);

    double origin_offset = config_.grid_size * config_.resolution / 2.0;
    std::vector<double> intensities(config_.grid_size * config_.grid_size, 0.0);
    std::vector<int> counts(config_.grid_size * config_.grid_size, 0);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(cloud_msg, *cloud);

    for (const auto& point : cloud->points) {
        int gx = static_cast<int>((point.x + origin_offset) / config_.resolution);
        int gy = static_cast<int>((point.y + origin_offset) / config_.resolution);
        if (gx >= 0 && gx < config_.grid_size && gy >= 0 && gy < config_.grid_size) {
            int idx = gy * config_.grid_size + gx;
            intensities[idx] += point.intensity;
            counts[idx]++;
        }
    }

    // Surface risk estimation from intensity (low intensity = potential slip/ice)
    for (int i = 0; i < config_.grid_size * config_.grid_size; ++i) {
        if (counts[i] > 0) {
            double avg_intensity = intensities[i] / counts[i];
            // Low intensity -> higher surface risk
            double surface_risk = std::max(0.0, 1.0 - avg_intensity / 100.0);
            int risk_cost = static_cast<int>(surface_risk * 100);
            grid_msg.data[i] = risk_cost;
        }
    }

    return grid_msg;
}

void HazardDetectorNode::enhanceCostmapWithMobilityRisk(nav_msgs::msg::OccupancyGrid& grid, 
                                                         const geometry_msgs::msg::TransformStamped& odom_to_base) {
    if (!mobility_risk_model_ || !mobility_risk_model_->isTrained()) return;

    // Get REAL terrain features from hazard detector
    auto terrain_features = hazard_detector_->getTerrainFeatures();
    if (terrain_features.empty()) return;

    double res = grid.info.resolution;
    double ox = grid.info.origin.position.x;
    double oy = grid.info.origin.position.y;
    int width = grid.info.width;
    int height = grid.info.height;

    double robot_x = odom_to_base.transform.translation.x;
    double robot_y = odom_to_base.transform.translation.y;

    // Get real rover state from fused odometry
    double robot_velocity = 0.0;
    double robot_angular_velocity = 0.0;
    {
        std::lock_guard<std::mutex> lock(fused_odom_mutex_);
        if (latest_fused_odom_) {
            robot_velocity = std::sqrt(
                latest_fused_odom_->twist.twist.linear.x * latest_fused_odom_->twist.twist.linear.x +
                latest_fused_odom_->twist.twist.linear.y * latest_fused_odom_->twist.twist.linear.y);
            robot_angular_velocity = latest_fused_odom_->twist.twist.angular.z;
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            int8_t hazard_val = grid.data[idx];
            
            if (hazard_val == -1) continue;
            
            // Use REAL terrain features from hazard detector
            const TerrainFeatures& base_features = terrain_features[idx];
            TerrainFeatures features = base_features;
            
            // Add REAL rover state from fused odometry
            features.velocity = robot_velocity;
            features.angular_velocity = robot_angular_velocity;
            features.wheel_torque = 0.0;  // Not available, set to 0
            features.sinkage = 0.0;
            
            auto pred = mobility_risk_model_->predict(features);
            double mobility_risk = pred.combined_risk();
            double uncertainty_penalty = pred.prediction_uncertainty * 100.0;  // Scale uncertainty to cost
            int mobility_cost = static_cast<int>(mobility_risk * 100);
            int uncertainty_cost = static_cast<int>(uncertainty_penalty);
            
            int blended_cost = static_cast<int>(
                hazard_weight_ * hazard_val + 
                mobility_risk_weight_ * mobility_cost +
                uncertainty_weight_ * uncertainty_cost
            );
            
            grid.data[idx] = std::min(100, std::max(0, blended_cost));
        }
    }
}

void HazardDetectorNode::thermalCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
        cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "mono16");
        std::lock_guard<std::mutex> lock(thermal_mutex_);
        latest_thermal_ = cv_ptr->image.clone();
    } catch (cv_bridge::Exception& e) {
        RCLCPP_WARN(this->get_logger(), "Thermal image conversion failed: %s", e.what());
    }
}

void HazardDetectorNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(thermal_mutex_);
    latest_camera_info_ = *msg;
    have_camera_info_ = true;
}

void HazardDetectorNode::timerCallback() {
    if (mobility_risk_model_ && mobility_risk_model_->isTrained()) {
    }
}

}  // namespace lunar_slam

RCLCPP_COMPONENTS_REGISTER_NODE(lunar_slam::HazardDetectorNode)