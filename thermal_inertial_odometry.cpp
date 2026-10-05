#include "lunar_slam/thermal_inertial_odometry.hpp"
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/eigen.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace lunar_slam {

ThermalInertialOdometry::ThermalInertialOdometry(const Config& config) : config_(config) {
    state_.covariance.setIdentity();
    state_.covariance *= 1e-3;
}

void ThermalInertialOdometry::processIMU(const sensor_msgs::msg::Imu::SharedPtr& msg) {
    double t = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
    imu_buffer_.push_back(msg);

    while (imu_buffer_.size() > 2 && 
           (t - (imu_buffer_.front()->header.stamp.sec + imu_buffer_.front()->header.stamp.nanosec * 1e-9)) > 1.0) {
        imu_buffer_.pop_front();
    }

    if (!state_.initialized || imu_buffer_.size() < 2) return;

    double t_prev = imu_buffer_[imu_buffer_.size() - 2]->header.stamp.sec + 
                    imu_buffer_[imu_buffer_.size() - 2]->header.stamp.nanosec * 1e-9;
    double dt = t - t_prev;
    if (dt <= 0 || dt > 0.1) return;

    updateIMUPreintegration(dt);
}

void ThermalInertialOdometry::processThermalImage(const sensor_msgs::msg::Image::SharedPtr& msg) {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO16);
    } catch (cv_bridge::Exception& e) {
        return;
    }

    cv::Mat frame_16 = cv_ptr->image;
    cv::Mat frame_8;
    double min_val, max_val;
    cv::minMaxLoc(frame_16, &min_val, &max_val);
    if (max_val - min_val > 0) {
        frame_16.convertTo(frame_8, CV_8UC1, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
    } else {
        frame_16.convertTo(frame_8, CV_8UC1, 0);
    }

    cv::Mat enhanced;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(frame_8, enhanced);

    double curr_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

    if (!state_.initialized) {
        initializeFirstFrame(enhanced);
        last_img_time_ = curr_time;
        return;
    }

    std::vector<cv::Point2f> curr_pts;
    std::vector<uchar> status;
    std::vector<float> err;
    trackFeatures(enhanced, curr_pts, status, err);

    std::vector<cv::Point2f> good_prev, good_curr;
    std::vector<int> good_ids;
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            good_prev.push_back(prev_pts_[i]);
            good_curr.push_back(curr_pts[i]);
            good_ids.push_back(track_ids_[i]);
        }
    }

    if (good_prev.size() >= 8) {
        cv::Mat essential = cv::findEssentialMat(good_curr, good_prev, 
            cv::Point2d(enhanced.cols/2, enhanced.rows/2), cv::RANSAC, 0.999, config_.ransac_threshold);
        cv::Mat R, t, mask;
        cv::recoverPose(essential, good_curr, good_prev, R, t, mask);

        Eigen::Matrix3d R_eig;
        Eigen::Vector3d t_eig;
        cv::cv2eigen(R, R_eig);
        cv::cv2eigen(t, t_eig);

        state_.imu_state.orientation = state_.imu_state.orientation * Eigen::Quaterniond(R_eig);
        
        // Use IMU-preintegrated translation magnitude to scale essential matrix translation
        // Essential matrix gives direction only; scale from IMU double integration
        double imu_trans_mag = state_.imu_state.velocity.norm() * 0.1; // approximate dt between frames
        double vision_trans_mag = t_eig.norm();
        double scale = (vision_trans_mag > 1e-6) ? imu_trans_mag / vision_trans_mag : 1.0;
        scale = std::max(0.1, std::min(10.0, scale)); // clamp
        
        state_.imu_state.position += state_.imu_state.orientation * t_eig * scale;
    }

    detectNewFeatures(enhanced, curr_pts);
    for (auto& pt : curr_pts) {
        if (prev_pts_.size() < config_.max_features) {
            prev_pts_.push_back(pt);
            track_ids_.push_back(next_track_id_++);
        }
    }

    prev_frame_ = enhanced.clone();
    prev_pts_ = curr_pts;
    last_img_time_ = curr_time;

    marginalizeOldFeatures();
}

void ThermalInertialOdometry::initializeFirstFrame(const cv::Mat& frame) {
    prev_frame_ = frame.clone();
    detectNewFeatures(frame, prev_pts_);
    track_ids_.resize(prev_pts_.size());
    std::iota(track_ids_.begin(), track_ids_.end(), 0);
    next_track_id_ = prev_pts_.size();
    state_.initialized = true;
    state_.imu_state.timestamp = last_img_time_;
}

void ThermalInertialOdometry::trackFeatures(const cv::Mat& curr_frame, 
                                             std::vector<cv::Point2f>& curr_pts,
                                             std::vector<uchar>& status,
                                             std::vector<float>& err) {
    std::vector<cv::Point2f> prev_pts = prev_pts_;
    cv::calcOpticalFlowPyrLK(prev_frame_, curr_frame, prev_pts, curr_pts, status, err,
                             cv::Size(21, 21), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));
}

void ThermalInertialOdometry::detectNewFeatures(const cv::Mat& frame, std::vector<cv::Point2f>& new_pts) {
    cv::Ptr<cv::ORB> orb = cv::ORB::create(config_.max_features);
    std::vector<cv::KeyPoint> keypoints;
    orb->detect(frame, keypoints);
    
    cv::KeyPointsFilter::retainBest(keypoints, config_.max_features);
    cv::KeyPointsFilter::removeDuplicated(keypoints, config_.feature_min_distance);
    
    cv::KeyPoint::convert(keypoints, new_pts);
}

void ThermalInertialOdometry::updateIMUPreintegration(double dt) {
    if (imu_buffer_.size() < 2) return;

    auto imu_curr = imu_buffer_.back();
    auto imu_prev = imu_buffer_[imu_buffer_.size() - 2];

    Eigen::Vector3d accel(imu_curr->linear_acceleration.x, imu_curr->linear_acceleration.y, imu_curr->linear_acceleration.z);
    Eigen::Vector3d gyro(imu_curr->angular_velocity.x, imu_curr->angular_velocity.y, imu_curr->angular_velocity.z);

    accel -= state_.imu_state.accel_bias;
    gyro -= state_.imu_state.gyro_bias;

    Eigen::Vector3d accel_world = state_.imu_state.orientation * accel + config_.gravity;
    
    // Correct integration order: position uses OLD velocity
    Eigen::Vector3d old_vel = state_.imu_state.velocity;
    state_.imu_state.position += old_vel * dt + 0.5 * accel_world * dt * dt;
    state_.imu_state.velocity += accel_world * dt;

    Eigen::Vector3d delta_theta = gyro * dt;
    double theta = delta_theta.norm();
    if (theta > 1e-8) {
        Eigen::Vector3d axis = delta_theta / theta;
        Eigen::AngleAxisd aa(theta, axis);
        state_.imu_state.orientation = state_.imu_state.orientation * Eigen::Quaterniond(aa);
    }
    state_.imu_state.orientation.normalize();

    state_.imu_state.timestamp = imu_curr->header.stamp.sec + imu_curr->header.stamp.nanosec * 1e-9;
}

void ThermalInertialOdometry::optimizePose() {
}

void ThermalInertialOdometry::marginalizeOldFeatures() {
    if (prev_pts_.size() > config_.max_features) {
        int remove_count = prev_pts_.size() - config_.max_features;
        prev_pts_.erase(prev_pts_.begin(), prev_pts_.begin() + remove_count);
        track_ids_.erase(track_ids_.begin(), track_ids_.begin() + remove_count);
    }
}

Eigen::Matrix3d ThermalInertialOdometry::skewSymmetric(const Eigen::Vector3d& v) const {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
         -v.y(), v.x(), 0;
    return m;
}

bool ThermalInertialOdometry::getPose(Eigen::Matrix4d& pose, double& timestamp) const {
    if (!state_.initialized) return false;
    pose = Eigen::Matrix4d::Identity();
    pose.block<3,3>(0,0) = state_.imu_state.orientation.toRotationMatrix();
    pose.block<3,1>(0,3) = state_.imu_state.position;
    timestamp = state_.imu_state.timestamp;
    return true;
}

ThermalInertialState ThermalInertialOdometry::getState() const {
    return state_;
}

}  // namespace lunar_slam