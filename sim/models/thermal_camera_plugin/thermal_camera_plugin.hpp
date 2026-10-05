#ifndef LUNAR_THERMAL_CAMERA_PLUGIN_HPP_
#define LUNAR_THERMAL_CAMERA_PLUGIN_HPP_

#include <gazebo/gazebo.hh>
#include <gazebo/sensors/sensors.hh>
#include <gazebo/rendering/rendering.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <thread>
#include <mutex>
#include <Eigen/Dense>

namespace gazebo {

class LunarThermalCameraPlugin : public SensorPlugin {
public:
    LunarThermalCameraPlugin();
    virtual ~LunarThermalCameraPlugin();

    void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf) override;

private:
    void OnNewFrame(const unsigned char* _image,
                    unsigned int _width, unsigned int _height,
                    unsigned int _depth, const std::string& _format);
    void PublishThermalImage();
    void PublishCameraInfo();
    double ComputeTemperatureFromWorld(const Eigen::Vector3d& world_pos);
    void AddThermalNoise(cv::Mat& image);
    void ApplyNonUniformityCorrection(cv::Mat& image);
    Eigen::Vector3d PixelToWorld(int u, int v);

    sensors::CameraSensorPtr camera_sensor_;
    rendering::CameraPtr camera_;
    event::ConnectionPtr new_frame_conn_;
    
    rclcpp::Node::SharedPtr ros_node_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr thermal_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
    rclcpp::executors::SingleThreadedExecutor executor_;
    std::thread ros_spin_thread_;
    std::mutex image_mutex_;
    cv::Mat latest_thermal_image_;
    bool new_image_available_ = false;

    // Camera intrinsics
    Eigen::Matrix3d K_;
    int image_width_ = 640;
    int image_height_ = 480;
    double fov_ = 1.047;
    std::string frame_id_ = "thermal_camera_link";

    // Simulation parameters
    double min_temp_ = 50.0;
    double max_temp_ = 400.0;
    double noise_sigma_ = 5.0;
    double nuc_sigma_ = 0.02;
    double ground_z_ = 0.0;  // Ground plane Z

    // Camera extrinsics (from camera to world)
    Eigen::Isometry3d camera_to_world_;
};

}  // namespace gazebo

#endif  // LUNAR_THERMAL_CAMERA_PLUGIN_HPP_
