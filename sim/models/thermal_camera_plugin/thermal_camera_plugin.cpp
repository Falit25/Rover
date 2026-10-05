#include "thermal_camera_plugin.hpp"
#include <gazebo/sensors/CameraSensor.hh>
#include <gazebo/rendering/Camera.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/transport/TransportTypes.hh>
#include <std_msgs/msg/header.hpp>
#include <cmath>

namespace gazebo {

GZ_REGISTER_SENSOR_PLUGIN(LunarThermalCameraPlugin)

LunarThermalCameraPlugin::LunarThermalCameraPlugin() : SensorPlugin() {}

LunarThermalCameraPlugin::~LunarThermalCameraPlugin() {
    executor_.cancel();

    if (ros_spin_thread_.joinable()) {
        ros_spin_thread_.join();
    }
}

void LunarThermalCameraPlugin::Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf) {
    SensorPlugin::Load(_sensor, _sdf);

    camera_sensor_ = std::dynamic_pointer_cast<sensors::CameraSensor>(_sensor);
    if (!camera_sensor_) {
        gzerr << "LunarThermalCameraPlugin: parent sensor is not a camera sensor\n";
        return;
    }

    camera_ = camera_sensor_->Camera();
    if (!camera_) {
        gzerr << "LunarThermalCameraPlugin: failed to get camera\n";
        return;
    }

    if (_sdf->HasElement("min_temperature")) min_temp_ = _sdf->Get<double>("min_temperature");
    if (_sdf->HasElement("max_temperature")) max_temp_ = _sdf->Get<double>("max_temperature");
    if (_sdf->HasElement("noise_sigma")) noise_sigma_ = _sdf->Get<double>("noise_sigma");
    if (_sdf->HasElement("nuc_sigma")) nuc_sigma_ = _sdf->Get<double>("nuc_sigma");
    if (_sdf->HasElement("frame_id")) frame_id_ = _sdf->Get<std::string>("frame_id");
    if (_sdf->HasElement("ground_z")) ground_z_ = _sdf->Get<double>("ground_z");

    image_width_ = camera_->ImageWidth();
    image_height_ = camera_->ImageHeight();
    fov_ = camera_->HFOV().Radian();

    // Compute camera intrinsics
    double fx = image_width_ / (2.0 * tan(fov_ / 2.0));
    double fy = fx;  // Assume square pixels
    double cx = image_width_ / 2.0;
    double cy = image_height_ / 2.0;
    K_ << fx, 0, cx,
          0, fy, cy,
          0, 0, 1;

    // Get camera pose in world frame
    ignition::math::Pose3d cam_pose = camera_->WorldPose();
    Eigen::Vector3d pos(cam_pose.Pos().X(), cam_pose.Pos().Y(), cam_pose.Pos().Z());
    Eigen::Quaterniond quat(cam_pose.Rot().W(), cam_pose.Rot().X(), cam_pose.Rot().Y(), cam_pose.Rot().Z());
    camera_to_world_ = Eigen::Isometry3d(quat);
    camera_to_world_.translation() = pos;

    new_frame_conn_ = camera_->ConnectNewImageFrame(
        std::bind(&LunarThermalCameraPlugin::OnNewFrame, this,
                  std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4,
                  std::placeholders::_5));

    if (!rclcpp::ok()) {
        rclcpp::init(0, nullptr);
    }

    ros_node_ = rclcpp::Node::make_shared("lunar_thermal_camera");
    thermal_pub_ = ros_node_->create_publisher<sensor_msgs::msg::Image>("/thermal/image_raw", 10);
    camera_info_pub_ = ros_node_->create_publisher<sensor_msgs::msg::CameraInfo>("/thermal/camera_info", 10);

    executor_.add_node(ros_node_);
    ros_spin_thread_ = std::thread([this]() {
        executor_.spin();
    });

    gzmsg << "LunarThermalCameraPlugin loaded successfully\n";
}

void LunarThermalCameraPlugin::OnNewFrame(const unsigned char* _image,
                                          unsigned int _width, unsigned int _height,
                                          unsigned int _depth, const std::string& _format) {
    if (_format != "L8" && _format != "R8G8B8" && _format != "B8G8R8") {
        return;
    }

    // Update camera pose from Gazebo (rover moves)
    ignition::math::Pose3d cam_pose = camera_->WorldPose();
    Eigen::Vector3d pos(cam_pose.Pos().X(), cam_pose.Pos().Y(), cam_pose.Pos().Z());
    Eigen::Quaterniond quat(cam_pose.Rot().W(), cam_pose.Rot().X(), cam_pose.Rot().Y(), cam_pose.Rot().Z());
    camera_to_world_ = Eigen::Isometry3d(quat);
    camera_to_world_.translation() = pos;

    cv::Mat color_image;
    if (_format == "L8") {
        color_image = cv::Mat(_height, _width, CV_8UC1, const_cast<unsigned char*>(_image));
    } else {
        color_image = cv::Mat(_height, _width, CV_8UC3, const_cast<unsigned char*>(_image));
        cv::cvtColor(color_image, color_image, cv::COLOR_BGR2GRAY);
    }

    cv::Mat thermal_16(image_height_, image_width_, CV_16UC1);
    
    for (int y = 0; y < image_height_; ++y) {
        for (int x = 0; x < image_width_; ++x) {
            double intensity = color_image.at<uchar>(y, x) / 255.0;
            double temp = min_temp_ + intensity * (max_temp_ - min_temp_);
            
            // Convert pixel to world coordinates
            Eigen::Vector3d world_pos = PixelToWorld(x, y);
            temp = ComputeTemperatureFromWorld(world_pos);
            
            thermal_16.at<uint16_t>(y, x) = static_cast<uint16_t>(temp * 100);
        }
    }

    AddThermalNoise(thermal_16);
    ApplyNonUniformityCorrection(thermal_16);

    {
        std::lock_guard<std::mutex> lock(image_mutex_);
        latest_thermal_image_ = thermal_16.clone();
        new_image_available_ = true;
    }

    PublishThermalImage();
    PublishCameraInfo();
}

Eigen::Vector3d LunarThermalCameraPlugin::PixelToWorld(int u, int v) {
    // Convert pixel to normalized camera coordinates
    double x = (u - K_(0,2)) / K_(0,0);
    double y = (v - K_(1,2)) / K_(1,1);
    
    // Ray direction in camera frame (z forward)
    Eigen::Vector3d ray_cam(x, y, 1.0);
    ray_cam.normalize();
    
    // Transform to world frame
    Eigen::Vector3d ray_world = camera_to_world_.linear() * ray_cam;
    Eigen::Vector3d cam_pos_world = camera_to_world_.translation();
    
    // Intersect with ground plane z = ground_z_
    double t = (ground_z_ - cam_pos_world.z()) / ray_world.z();
    if (t <= 0) {
        // Ray points upward or parallel, return camera position projected to ground
        return Eigen::Vector3d(cam_pos_world.x(), cam_pos_world.y(), ground_z_);
    }
    
    Eigen::Vector3d world_pos = cam_pos_world + t * ray_world;
    world_pos.z() = ground_z_;
    return world_pos;
}

double LunarThermalCameraPlugin::ComputeTemperatureFromWorld(const Eigen::Vector3d& world_pos) {
    double base_temp = 250.0;
    double x = world_pos.x();
    double y = world_pos.y();
    
    // Crater at (10, 10)
    double crater_effect = 0.0;
    double cx = 10.0, cy = 10.0;
    double dist = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
    if (dist < 10.0) {
        crater_effect = -50.0 * std::exp(-dist * dist / 20.0);
    }

    // Cold trap at (-10, -15) to (-5, -10)
    double cold_trap_effect = 0.0;
    if (x > -15 && x < -5 && y > -20 && y < -10) {
        cold_trap_effect = -100.0;
    }

    // Rock field at (5, -10) to (13, -2)
    double rock_effect = 0.0;
    if (x > 1 && x < 9 && y > -14 && y < -6) {
        rock_effect = 30.0;
    }

    double temp = base_temp + crater_effect + cold_trap_effect + rock_effect;
    return std::max(min_temp_, std::min(max_temp_, temp));
}

void LunarThermalCameraPlugin::AddThermalNoise(cv::Mat& image) {
    cv::Mat noise(image.size(), CV_16SC1);
    cv::randn(noise, 0, noise_sigma_ * 100);
    image = image + noise;
}

void LunarThermalCameraPlugin::ApplyNonUniformityCorrection(cv::Mat& image) {
    // Apply multiplicative non-uniformity noise (no normalization!)
    cv::Mat nuc(image.size(), CV_32FC1);
    cv::randn(nuc, 1.0, nuc_sigma_);
    
    cv::Mat float_image;
    image.convertTo(float_image, CV_32FC1);
    float_image = float_image.mul(nuc);
    
    // Clip to valid range without normalization
    cv::threshold(float_image, float_image, min_temp_ * 100, min_temp_ * 100, cv::THRESH_TOZERO);
    cv::threshold(float_image, float_image, max_temp_ * 100, max_temp_ * 100, cv::THRESH_TRUNC);
    
    float_image.convertTo(image, CV_16UC1);
}

void LunarThermalCameraPlugin::PublishThermalImage() {
    cv::Mat image_to_publish;
    {
        std::lock_guard<std::mutex> lock(image_mutex_);
        if (!new_image_available_) return;
        image_to_publish = latest_thermal_image_.clone();
        new_image_available_ = false;
    }

    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono16", image_to_publish).toImageMsg();
    msg->header.frame_id = frame_id_;
    msg->header.stamp = ros_node_->now();
    thermal_pub_->publish(*msg);
}

void LunarThermalCameraPlugin::PublishCameraInfo() {
    sensor_msgs::msg::CameraInfo info;
    info.header.frame_id = frame_id_;
    info.header.stamp = ros_node_->now();
    info.width = image_width_;
    info.height = image_height_;
    info.distortion_model = "plumb_bob";
    info.d = {0, 0, 0, 0, 0};
    info.k = {K_(0,0), K_(0,1), K_(0,2),
              K_(1,0), K_(1,1), K_(1,2),
              K_(2,0), K_(2,1), K_(2,2)};
    info.r = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    info.p = {K_(0,0), K_(0,1), K_(0,2), 0,
              K_(1,0), K_(1,1), K_(1,2), 0,
              K_(2,0), K_(2,1), K_(2,2), 0};
    info.binning_x = 0;
    info.binning_y = 0;
    info.roi.x_offset = 0;
    info.roi.y_offset = 0;
    info.roi.width = 0;
    info.roi.height = 0;
    info.roi.do_rectify = false;
    
    camera_info_pub_->publish(info);
}

}  // namespace gazebo