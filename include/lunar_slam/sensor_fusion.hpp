#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <deque>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

namespace lunar_slam {

struct NominalState {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    double timestamp = 0.0;
};

struct ErrorStateCovariance {
    Eigen::Matrix<double, 15, 15> P = Eigen::Matrix<double, 15, 15>::Identity() * 1e-3;
};

struct SensorFusionConfig {
    double imu_rate = 200.0;
    double lidar_rate = 10.0;
    double thermal_rate = 10.0;
    double wheel_rate = 50.0;
    Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -1.62);
    Eigen::Matrix<double, 6, 6> imu_noise = Eigen::Matrix<double, 6, 6>::Identity() * 1e-4;
    Eigen::Matrix<double, 6, 6> lidar_noise = Eigen::Matrix<double, 6, 6>::Identity() * 1e-2;
    Eigen::Matrix<double, 6, 6> thermal_noise = Eigen::Matrix<double, 6, 6>::Identity() * 5e-2;
    Eigen::Matrix<double, 3, 3> wheel_noise = Eigen::Matrix<double, 3, 3>::Identity() * 1e-3;
    double bias_random_walk_accel = 1e-5;
    double bias_random_walk_gyro = 1e-6;
};

class SensorFusionEKF {
public:
    explicit SensorFusionEKF(const SensorFusionConfig& config = SensorFusionConfig());

    void predict(double dt);
    void updateIMU(const sensor_msgs::msg::Imu::SharedPtr& msg);
    void updateLidarOdom(const nav_msgs::msg::Odometry::SharedPtr& msg);
    void updateThermalOdom(const nav_msgs::msg::Odometry::SharedPtr& msg);
    void updateWheelOdom(const nav_msgs::msg::Odometry::SharedPtr& msg);
    bool getFusedPose(Eigen::Matrix4d& pose, Eigen::Matrix<double, 6, 6>& cov, double& timestamp) const;
    NominalState getNominalState() const { return nominal_; }
    ErrorStateCovariance getCovariance() const { return cov_; }

private:
    SensorFusionConfig config_;
    NominalState nominal_;
    ErrorStateCovariance cov_;
    std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_buffer_;
    double last_predict_time_ = 0.0;

    // Error-state indices
    static constexpr int IDX_DP = 0;
    static constexpr int IDX_DV = 3;
    static constexpr int IDX_DTHETA = 6;
    static constexpr int IDX_DBA = 9;
    static constexpr int IDX_DBG = 12;

    Eigen::Matrix<double, 15, 15> computeF(double dt) const;
    Eigen::Matrix<double, 15, 12> computeG(double dt) const;
    Eigen::Matrix<double, 15, 15> computeQ(double dt) const;
    void predictStep(double dt);
    void measurementUpdate(const Eigen::VectorXd& z, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R);
    Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) const;
    Eigen::Matrix<double, 6, 15> measurementJacobianPosVel() const;
    Eigen::Matrix<double, 6, 15> measurementJacobianPose() const;
    Eigen::VectorXd nominalToMeasurementPosVel() const;
    Eigen::VectorXd nominalToMeasurementPose() const;
    void injectErrorState(const Eigen::VectorXd& dx);
};

}  // namespace lunar_slam