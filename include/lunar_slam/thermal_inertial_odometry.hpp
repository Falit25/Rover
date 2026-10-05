#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace lunar_slam {

struct IMUState {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    double timestamp = 0.0;
};

struct ThermalFeature {
    cv::Point2f px;
    Eigen::Vector3d position_3d;
    int track_id = -1;
    int age = 0;
    float response = 0.0f;
};

struct ThermalInertialState {
    IMUState imu_state;
    std::vector<ThermalFeature> features;
    Eigen::Matrix<double, 15, 15> covariance = Eigen::Matrix<double, 15, 15>::Identity() * 1e-3;
    bool initialized = false;
};

class ThermalInertialOdometry {
public:
    struct Config {
        double img_rate = 10.0;
        double imu_rate = 200.0;
        int max_features = 500;
        double feature_min_distance = 15.0;
        double ransac_threshold = 2.0;
        double parallax_threshold = 10.0;
        Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -1.62);
        double accel_noise = 1e-3;
        double gyro_noise = 1e-4;
        double accel_bias_noise = 1e-5;
        double gyro_bias_noise = 1e-6;
    };

    explicit ThermalInertialOdometry(const Config& config = Config());

    void processIMU(const sensor_msgs::msg::Imu::SharedPtr& msg);
    void processThermalImage(const sensor_msgs::msg::Image::SharedPtr& msg);
    bool getPose(Eigen::Matrix4d& pose, double& timestamp) const;
    ThermalInertialState getState() const;

private:
    Config config_;
    ThermalInertialState state_;
    cv::Mat prev_frame_;
    std::vector<cv::Point2f> prev_pts_;
    std::vector<int> track_ids_;
    int next_track_id_ = 0;
    std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_buffer_;
    double last_img_time_ = 0.0;

    void initializeFirstFrame(const cv::Mat& frame);
    void trackFeatures(const cv::Mat& curr_frame, std::vector<cv::Point2f>& curr_pts,
                       std::vector<uchar>& status, std::vector<float>& err);
    void detectNewFeatures(const cv::Mat& frame, std::vector<cv::Point2f>& new_pts);
    void updateIMUPreintegration(double dt);
    void optimizePose();
    void marginalizeOldFeatures();
    Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) const;
};

}  // namespace lunar_slam