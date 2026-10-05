#ifndef LUNAR_SLAM_PATH_PLANNER_HPP_
#define LUNAR_SLAM_PATH_PLANNER_HPP_

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

namespace lunar_slam {

class PathPlanner {
public:
    PathPlanner() = default;

    nav_msgs::msg::Path planSafePath(const nav_msgs::msg::OccupancyGrid& grid,
                                     const geometry_msgs::msg::Pose& start_pose,
                                     const geometry_msgs::msg::Pose& goal_pose);
};

} // namespace lunar_slam

#endif