#include "lunar_slam/sensor_fusion.hpp"
#include <cmath>

namespace lunar_slam {

SensorFusionEKF::SensorFusionEKF(const SensorFusionConfig& config) : config_(config) {
    cov_.P.setIdentity();
    cov_.P *= 1e-3;
}

void SensorFusionEKF::predict(double dt) {
    if (dt <= 0 || dt > 1.0) return;
    predictStep(dt);
}

void SensorFusionEKF::updateIMU(const sensor_msgs::msg::Imu::SharedPtr& msg) {
    double t = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    imu_buffer_.push_back(msg);

    while (imu_buffer_.size() > 2 && 
           (t - (imu_buffer_.front()->header.stamp.sec + imu_buffer_.front()->header.stamp.nanosec * 1e-9)) > 0.5) {
        imu_buffer_.pop_front();
    }

    if (last_predict_time_ > 0 && imu_buffer_.size() >= 2) {
        double dt = t - last_predict_time_;
        if (dt > 0 && dt < 0.1) {
            predict(dt);
        }
    }
    last_predict_time_ = t;
}

void SensorFusionEKF::updateLidarOdom(const nav_msgs::msg::Odometry::SharedPtr& msg) {
    Eigen::VectorXd z = nominalToMeasurementPose();
    Eigen::VectorXd z_meas(6);
    z_meas << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z,
              0, 0, 0;
    
    Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                         msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0);
    z_meas(3) = euler.z();
    z_meas(4) = euler.y();
    z_meas(5) = euler.x();

    Eigen::VectorXd y = z_meas - z;
    y(3) = std::atan2(std::sin(y(3)), std::cos(y(3)));
    y(4) = std::atan2(std::sin(y(4)), std::cos(y(4)));
    y(5) = std::atan2(std::sin(y(5)), std::cos(y(5)));

    Eigen::MatrixXd H = measurementJacobianPose();
    measurementUpdate(y, H, config_.lidar_noise);
}

void SensorFusionEKF::updateThermalOdom(const nav_msgs::msg::Odometry::SharedPtr& msg) {
    Eigen::VectorXd z = nominalToMeasurementPose();
    Eigen::VectorXd z_meas(6);
    z_meas << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z,
              0, 0, 0;
    
    Eigen::Quaterniond q(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                         msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0);
    z_meas(3) = euler.z();
    z_meas(4) = euler.y();
    z_meas(5) = euler.x();

    Eigen::VectorXd y = z_meas - z;
    y(3) = std::atan2(std::sin(y(3)), std::cos(y(3)));
    y(4) = std::atan2(std::sin(y(4)), std::cos(y(4)));
    y(5) = std::atan2(std::sin(y(5)), std::cos(y(5)));

    Eigen::MatrixXd H = measurementJacobianPose();
    measurementUpdate(y, H, config_.thermal_noise);
}

void SensorFusionEKF::updateWheelOdom(const nav_msgs::msg::Odometry::SharedPtr& msg) {
    // Wheel odometry provides velocity, not absolute position
    Eigen::VectorXd z_meas(3);
    z_meas << msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z;

    Eigen::VectorXd y = z_meas - nominal_.velocity;

    Eigen::MatrixXd H(3, 15);
    H.setZero();
    H.block<3, 3>(0, IDX_DV) = Eigen::Matrix3d::Identity();

    measurementUpdate(y, H, config_.wheel_noise);
}

bool SensorFusionEKF::getFusedPose(Eigen::Matrix4d& pose, Eigen::Matrix<double, 6, 6>& cov, double& timestamp) const {
    if (nominal_.timestamp == 0) return false;
    pose = Eigen::Matrix4d::Identity();
    pose.block<3,3>(0,0) = nominal_.orientation.toRotationMatrix();
    pose.block<3,1>(0,3) = nominal_.position;

    cov = cov_.P.block<6,6>(0,0);
    timestamp = nominal_.timestamp;
    return true;
}

Eigen::Matrix<double, 15, 15> SensorFusionEKF::computeF(double dt) const {
    Eigen::Matrix<double, 15, 15> F = Eigen::Matrix<double, 15, 15>::Identity();
    Eigen::Matrix3d R = nominal_.orientation.toRotationMatrix();
    Eigen::Vector3d accel_world = R * (Eigen::Vector3d::Zero() - nominal_.accel_bias) + config_.gravity;

    F.block<3,3>(IDX_DP, IDX_DV) = Eigen::Matrix3d::Identity() * dt;
    F.block<3,3>(IDX_DP, IDX_DTHETA) = -R * skewSymmetric(accel_world) * dt * dt * 0.5;
    F.block<3,3>(IDX_DP, IDX_DBA) = -R * dt * dt * 0.5;
    
    F.block<3,3>(IDX_DV, IDX_DTHETA) = -R * skewSymmetric(accel_world) * dt;
    F.block<3,3>(IDX_DV, IDX_DBA) = -R * dt;
    
    F.block<3,3>(IDX_DTHETA, IDX_DBG) = -Eigen::Matrix3d::Identity() * dt;

    return F;
}

Eigen::Matrix<double, 15, 12> SensorFusionEKF::computeG(double dt) const {
    Eigen::Matrix<double, 15, 12> G = Eigen::Matrix<double, 15, 12>::Zero();
    Eigen::Matrix3d R = nominal_.orientation.toRotationMatrix();

    G.block<3,3>(IDX_DV, 0) = R * dt;
    G.block<3,3>(IDX_DTHETA, 3) = R * dt;
    G.block<3,3>(IDX_DBA, 6) = Eigen::Matrix3d::Identity() * dt;
    G.block<3,3>(IDX_DBG, 9) = Eigen::Matrix3d::Identity() * dt;

    return G;
}

Eigen::Matrix<double, 15, 15> SensorFusionEKF::computeQ(double dt) const {
    Eigen::Matrix<double, 12, 12> Qc = Eigen::Matrix<double, 12, 12>::Zero();
    Qc.block<3,3>(0,0) = config_.imu_noise.block<3,3>(0,0);
    Qc.block<3,3>(3,3) = config_.imu_noise.block<3,3>(3,3);
    Qc.block<3,3>(6,6) = Eigen::Matrix3d::Identity() * config_.bias_random_walk_accel;
    Qc.block<3,3>(9,9) = Eigen::Matrix3d::Identity() * config_.bias_random_walk_gyro;

    Eigen::Matrix<double, 15, 12> G = computeG(dt);
    return G * Qc * G.transpose() * dt;
}

void SensorFusionEKF::predictStep(double dt) {
    Eigen::Matrix<double, 15, 15> F = computeF(dt);
    Eigen::Matrix<double, 15, 15> Q = computeQ(dt);

    Eigen::Vector3d accel_meas(0, 0, 0);
    Eigen::Vector3d gyro_meas(0, 0, 0);

    if (!imu_buffer_.empty()) {
        auto imu = imu_buffer_.back();
        accel_meas << imu->linear_acceleration.x, imu->linear_acceleration.y, imu->linear_acceleration.z;
        gyro_meas << imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z;
    }

    Eigen::Vector3d accel_unbiased = accel_meas - nominal_.accel_bias;
    Eigen::Vector3d gyro_unbiased = gyro_meas - nominal_.gyro_bias;

    Eigen::Vector3d accel_world = nominal_.orientation.toRotationMatrix() * accel_unbiased + config_.gravity;
    
    // Nominal state propagation
    nominal_.position += nominal_.velocity * dt + 0.5 * accel_world * dt * dt;
    nominal_.velocity += accel_world * dt;
    
    Eigen::Vector3d delta_theta = gyro_unbiased * dt;
    double theta = delta_theta.norm();
    if (theta > 1e-8) {
        Eigen::Vector3d axis = delta_theta / theta;
        Eigen::AngleAxisd aa(theta, axis);
        nominal_.orientation = nominal_.orientation * Eigen::Quaterniond(aa);
    }
    nominal_.orientation.normalize();

    nominal_.timestamp += dt;
    
    // Error-state covariance propagation
    cov_.P = F * cov_.P * F.transpose() + Q;
}

void SensorFusionEKF::measurementUpdate(const Eigen::VectorXd& y, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R) {
    Eigen::MatrixXd S = H * cov_.P * H.transpose() + R;
    Eigen::MatrixXd K = cov_.P * H.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));

    Eigen::VectorXd dx = K * y;
    injectErrorState(dx);

    cov_.P = (Eigen::Matrix<double, 15, 15>::Identity() - K * H) * cov_.P;
}

Eigen::Matrix3d SensorFusionEKF::skewSymmetric(const Eigen::Vector3d& v) const {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
         -v.y(), v.x(), 0;
    return m;
}

Eigen::VectorXd SensorFusionEKF::nominalToMeasurementPosVel() const {
    Eigen::VectorXd z(3);
    z = nominal_.position;
    return z;
}

Eigen::VectorXd SensorFusionEKF::nominalToMeasurementPose() const {
    Eigen::VectorXd z(6);
    z.head(3) = nominal_.position;
    Eigen::Vector3d euler = nominal_.orientation.toRotationMatrix().eulerAngles(2, 1, 0);
    z(3) = euler.z();
    z(4) = euler.y();
    z(5) = euler.x();
    return z;
}

Eigen::Matrix<double, 6, 15> SensorFusionEKF::measurementJacobianPosVel() const {
    Eigen::Matrix<double, 6, 15> H = Eigen::Matrix<double, 6, 15>::Zero();
    H.block<3,3>(0, IDX_DP) = Eigen::Matrix3d::Identity();
    H.block<3,3>(3, IDX_DV) = Eigen::Matrix3d::Identity();
    return H;
}

Eigen::Matrix<double, 6, 15> SensorFusionEKF::measurementJacobianPose() const {
    Eigen::Matrix<double, 6, 15> H = Eigen::Matrix<double, 6, 15>::Zero();
    H.block<3,3>(0, IDX_DP) = Eigen::Matrix3d::Identity();
    H.block<3,3>(3, IDX_DTHETA) = Eigen::Matrix3d::Identity();
    return H;
}

void SensorFusionEKF::injectErrorState(const Eigen::VectorXd& dx) {
    // Position correction
    nominal_.position += dx.segment<3>(IDX_DP);
    
    // Velocity correction
    nominal_.velocity += dx.segment<3>(IDX_DV);
    
    // Orientation correction (small angle)
    Eigen::Vector3d dtheta = dx.segment<3>(IDX_DTHETA);
    if (dtheta.norm() > 1e-10) {
        Eigen::Quaterniond dq = Eigen::Quaterniond(
            1.0, dtheta.x() * 0.5, dtheta.y() * 0.5, dtheta.z() * 0.5);
        dq.normalize();
        nominal_.orientation = nominal_.orientation * dq;
        nominal_.orientation.normalize();
    }
    
    // Bias corrections
    nominal_.accel_bias += dx.segment<3>(IDX_DBA);
    nominal_.gyro_bias += dx.segment<3>(IDX_DBG);
}

}  // namespace lunar_slam