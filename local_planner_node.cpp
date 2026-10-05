#include "lunar_slam/local_planner_node.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>
#include <algorithm>

namespace lunar_slam {

LocalPlannerNode::LocalPlannerNode(const rclcpp::NodeOptions& options)
    : Node("local_planner_node", options) {

    this->declare_parameter("global_path_topic", "/global_path");
    this->declare_parameter("costmap_topic", "/traversability_costmap");
    this->declare_parameter("cmd_vel_topic", "/cmd_vel");
    this->declare_parameter("local_path_topic", "/local_path");
    this->declare_parameter("timer_rate", 20.0);

    config_.max_vel_x = this->declare_parameter("local_planner.max_vel_x", 0.5);
    config_.min_vel_x = this->declare_parameter("local_planner.min_vel_x", -0.2);
    config_.max_vel_theta = this->declare_parameter("local_planner.max_vel_theta", 1.0);
    config_.acc_lim_x = this->declare_parameter("local_planner.acc_lim_x", 0.5);
    config_.acc_lim_theta = this->declare_parameter("local_planner.acc_lim_theta", 1.5);
    config_.sim_time = this->declare_parameter("local_planner.sim_time", 2.0);
    config_.sim_granularity = this->declare_parameter("local_planner.sim_granularity", 0.05);
    config_.vx_samples = this->declare_parameter("local_planner.vx_samples", 10);
    config_.vtheta_samples = this->declare_parameter("local_planner.vtheta_samples", 20);
    config_.path_distance_bias = this->declare_parameter("local_planner.path_distance_bias", 0.6);
    config_.goal_distance_bias = this->declare_parameter("local_planner.goal_distance_bias", 0.4);
    config_.occdist_scale = this->declare_parameter("local_planner.occdist_scale", 0.5);
    config_.risk_scale = this->declare_parameter("local_planner.risk_scale", 2.0);

    std::string global_path_topic = this->get_parameter("global_path_topic").as_string();
    std::string costmap_topic = this->get_parameter("costmap_topic").as_string();
    std::string cmd_vel_topic = this->get_parameter("cmd_vel_topic").as_string();
    std::string local_path_topic = this->get_parameter("local_path_topic").as_string();
    double timer_rate = this->get_parameter("timer_rate").as_double();

    global_path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
        global_path_topic, 10, std::bind(&LocalPlannerNode::globalPathCallback, this, std::placeholders::_1));

    costmap_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        costmap_topic, 10, std::bind(&LocalPlannerNode::costmapCallback, this, std::placeholders::_1));

    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic, 10);
    local_path_pub_ = this->create_publisher<nav_msgs::msg::Path>(local_path_topic, 10);

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / timer_rate),
        std::bind(&LocalPlannerNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "Local Planner Node initialized");
}

void LocalPlannerNode::globalPathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_global_path_ = msg;
    current_waypoint_idx_ = 0;
}

void LocalPlannerNode::costmapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_costmap_ = msg;
}

geometry_msgs::msg::Pose2D LocalPlannerNode::getRobotPose() {
    geometry_msgs::msg::Pose2D pose;
    try {
        geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
            "odom", "base_link", tf2::TimePointZero);
        pose.x = transform.transform.translation.x;
        pose.y = transform.transform.translation.y;
        tf2::Quaternion q(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        pose.theta = yaw;
    } catch (tf2::TransformException& ex) {
        RCLCPP_WARN(this->get_logger(), "TF transform failed: %s", ex.what());
        pose.x = pose.y = pose.theta = 0.0;
    }
    return pose;
}

void LocalPlannerNode::timerCallback() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    
    if (!latest_global_path_ || !latest_costmap_ || latest_global_path_->poses.empty()) {
        return;
    }

    geometry_msgs::msg::Pose2D robot_pose = getRobotPose();
    
    while (current_waypoint_idx_ + 1 < latest_global_path_->poses.size()) {
        double wx = latest_global_path_->poses[current_waypoint_idx_].pose.position.x;
        double wy = latest_global_path_->poses[current_waypoint_idx_].pose.position.y;
        double dist = std::hypot(wx - robot_pose.x, wy - robot_pose.y);
        if (dist < 0.5) {
            current_waypoint_idx_++;
        } else {
            break;
        }
    }

    if (current_waypoint_idx_ >= latest_global_path_->poses.size()) {
        geometry_msgs::msg::Twist cmd;
        cmd_vel_pub_->publish(cmd);
        return;
    }

    Trajectory best_traj;
    best_traj.cost = std::numeric_limits<double>::infinity();

    for (int i = 0; i <= config_.vx_samples; ++i) {
        double vx = config_.min_vel_x + (config_.max_vel_x - config_.min_vel_x) * i / config_.vx_samples;
        for (int j = 0; j <= config_.vtheta_samples; ++j) {
            double vtheta = -config_.max_vel_theta + 2.0 * config_.max_vel_theta * j / config_.vtheta_samples;
            Trajectory traj = evaluateTrajectory(vx, vtheta);
            if (traj.valid && traj.cost < best_traj.cost) {
                best_traj = traj;
            }
        }
    }

    if (best_traj.valid && !best_traj.poses.empty()) {
        geometry_msgs::msg::Twist cmd;
        double dt = config_.sim_granularity;
        cmd.linear.x = (best_traj.poses[1].x - best_traj.poses[0].x) / dt;
        cmd.angular.z = (best_traj.poses[1].theta - best_traj.poses[0].theta) / dt;
        
        // Clamp velocities
        cmd.linear.x = std::clamp(cmd.linear.x, config_.min_vel_x, config_.max_vel_x);
        cmd.angular.z = std::clamp(cmd.angular.z, -config_.max_vel_theta, config_.max_vel_theta);
        
        cmd_vel_pub_->publish(cmd);

        nav_msgs::msg::Path local_path;
        local_path.header.frame_id = "odom";
        local_path.header.stamp = this->now();
        for (const auto& p : best_traj.poses) {
            geometry_msgs::msg::PoseStamped ps;
            ps.header = local_path.header;
            ps.pose.position.x = p.x;
            ps.pose.position.y = p.y;
            ps.pose.position.z = 0.0;
            tf2::Quaternion q;
            q.setRPY(0, 0, p.theta);
            ps.pose.orientation.x = q.x();
            ps.pose.orientation.y = q.y();
            ps.pose.orientation.z = q.z();
            ps.pose.orientation.w = q.w();
            local_path.poses.push_back(ps);
        }
        local_path_pub_->publish(local_path);
    }
}

LocalPlannerNode::Trajectory LocalPlannerNode::evaluateTrajectory(double vx, double vtheta) {
    Trajectory traj;
    traj.poses.reserve(config_.sim_time / config_.sim_granularity + 1);
    
    geometry_msgs::msg::Pose2D pose = getRobotPose();
    traj.poses.push_back(pose);

    double dt = config_.sim_granularity;
    for (double t = 0; t < config_.sim_time; t += dt) {
        pose.x += vx * std::cos(pose.theta) * dt;
        pose.y += vx * std::sin(pose.theta) * dt;
        pose.theta += vtheta * dt;
        traj.poses.push_back(pose);
    }

    traj.cost = computeTrajectoryCost(traj);
    traj.valid = traj.cost < 1e6;

    return traj;
}

double LocalPlannerNode::computeTrajectoryCost(const Trajectory& traj) {
    if (!latest_costmap_) return 1e6;

    double path_cost = 0.0;
    double goal_cost = 0.0;
    double occ_cost = 0.0;
    double risk_cost = 0.0;

    if (latest_global_path_ && latest_global_path_->poses.size() > current_waypoint_idx_) {
        double goal_x = latest_global_path_->poses.back().pose.position.x;
        double goal_y = latest_global_path_->poses.back().pose.position.y;
        
        for (const auto& p : traj.poses) {
            double min_dist = 1e6;
            for (size_t i = current_waypoint_idx_; i < latest_global_path_->poses.size(); ++i) {
                double wx = latest_global_path_->poses[i].pose.position.x;
                double wy = latest_global_path_->poses[i].pose.position.y;
                double d = std::hypot(p.x - wx, p.y - wy);
                min_dist = std::min(min_dist, d);
            }
            path_cost += min_dist;
        }

        if (!traj.poses.empty()) {
            const auto& last = traj.poses.back();
            goal_cost = std::hypot(last.x - goal_x, last.y - goal_y);
        }
    }

    if (latest_costmap_) {
        double res = latest_costmap_->info.resolution;
        double origin_x = latest_costmap_->info.origin.position.x;
        double origin_y = latest_costmap_->info.origin.position.y;
        int width = latest_costmap_->info.width;
        int height = latest_costmap_->info.height;

        for (const auto& p : traj.poses) {
            int gx = static_cast<int>((p.x - origin_x) / res);
            int gy = static_cast<int>((p.y - origin_y) / res);
            
            if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
                int idx = gy * width + gx;
                int8_t cell = latest_costmap_->data[idx];
                if (cell == -1) {
                    // Unknown terrain = invalid trajectory
                    return 1e6;
                }
                if (cell >= 100) {
                    occ_cost += 1000.0;
                } else if (cell >= 80) {
                    occ_cost += 100.0;
                } else if (cell >= 50) {
                    occ_cost += 10.0;
                }
                if (cell > 0) {
                    risk_cost += cell / 100.0;
                }
            } else {
                occ_cost += 500.0;
            }
        }
    }

    return config_.path_distance_bias * path_cost +
           config_.goal_distance_bias * goal_cost +
           config_.occdist_scale * occ_cost +
           config_.risk_scale * risk_cost;
}

}  // namespace lunar_slam

RCLCPP_COMPONENTS_REGISTER_NODE(lunar_slam::LocalPlannerNode)