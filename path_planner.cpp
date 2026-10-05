#include "lunar_slam/path_planner.hpp"
#include <cmath>
#include <queue>
#include <vector>

namespace lunar_slam {

struct Node {
    int x, y;
    double g_cost, h_cost;
    int parent_x, parent_y;

    double fCost() const { return g_cost + h_cost; }
    bool operator>(const Node& other) const { return fCost() > other.fCost(); }
};
nav_msgs::msg::Path PathPlanner::planSafePath(const nav_msgs::msg::OccupancyGrid& grid,
                                              const geometry_msgs::msg::Pose& start_pose,
                                              const geometry_msgs::msg::Pose& goal_pose) {
    nav_msgs::msg::Path path_msg;
    path_msg.header = grid.header;

    int width = grid.info.width;
    int height = grid.info.height;
    double res = grid.info.resolution;
    double origin_x = grid.info.origin.position.x;
    double origin_y = grid.info.origin.position.y;

    // Convert start and goal coordinates into grid indices
    int start_x = static_cast<int>((start_pose.position.x - origin_x) / res);
    int start_y = static_cast<int>((start_pose.position.y - origin_y) / res);
    int goal_x = static_cast<int>((goal_pose.position.x - origin_x) / res);
    int goal_y = static_cast<int>((goal_pose.position.y - origin_y) / res);

    // Bounds check
    if (start_x < 0 || start_x >= width || start_y < 0 || start_y >= height ||
        goal_x < 0 || goal_x >= width || goal_y < 0 || goal_y >= height) {
        return path_msg;
    }

    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_set;
    std::vector<std::vector<bool>> closed_set(width, std::vector<bool>(height, false));

    auto heuristic = [](int x1, int y1, int x2, int y2) {
        return std::hypot(x1 - x2, y1 - y2);
    };

    open_set.push({start_x, start_y, 0.0, heuristic(start_x, start_y, goal_x, goal_y), -1, -1});

    // Simple Forward Waypoint Construction
    geometry_msgs::msg::PoseStamped wp_start, wp_goal;
    wp_start.header = grid.header;
    wp_start.pose = start_pose;
    
    wp_goal.header = grid.header;
    wp_goal.pose = goal_pose;

    path_msg.poses.push_back(wp_start);
    path_msg.poses.push_back(wp_goal);

    return path_msg;
}

} // namespace lunar_slam