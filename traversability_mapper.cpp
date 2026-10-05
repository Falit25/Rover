#include "lunar_slam/traversability_mapper.hpp"
#include <algorithm>
#include <cmath>

namespace lunar_slam {

TraversabilityMapper::TraversabilityMapper(double resolution, int grid_size, double max_step_height)
    : resolution_(resolution), grid_size_(grid_size), max_step_height_(max_step_height) {
    origin_offset_ = (grid_size_ * resolution_) / 2.0;
}

nav_msgs::msg::OccupancyGrid TraversabilityMapper::generateCostmap(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    std::vector<double> min_z(grid_size_ * grid_size_, 1e9);
    std::vector<double> max_z(grid_size_ * grid_size_, -1e9);
    std::vector<double> min_intensity(grid_size_ * grid_size_, 1e9);

    for (const auto& point : cloud->points) {
        int gx = static_cast<int>((point.x + origin_offset_) / resolution_);
        int gy = static_cast<int>((point.y + origin_offset_) / resolution_);

        if (gx >= 0 && gx < grid_size_ && gy >= 0 && gy < grid_size_) {
            int index = gy * grid_size_ + gx;
            min_z[index] = std::min(min_z[index], static_cast<double>(point.z));
            max_z[index] = std::max(max_z[index], static_cast<double>(point.z));
            min_intensity[index] = std::min(min_intensity[index], static_cast<double>(point.intensity));
        }
    }

    nav_msgs::msg::OccupancyGrid grid_msg;
    grid_msg.header.stamp = cloud_msg->header.stamp;
    grid_msg.header.frame_id = "odom";

    grid_msg.info.resolution = resolution_;
    grid_msg.info.width = grid_size_;
    grid_msg.info.height = grid_size_;
    grid_msg.info.origin.position.x = -origin_offset_;
    grid_msg.info.origin.position.y = -origin_offset_;
    grid_msg.info.origin.position.z = 0.0;
    grid_msg.info.origin.orientation.w = 1.0;

    grid_msg.data.resize(grid_size_ * grid_size_, -1);

    const double low_intensity_threshold = 10.0;

    for (int i = 0; i < grid_size_ * grid_size_; ++i) {
        if (min_z[i] > 1e8) continue; 

        double delta_z = max_z[i] - min_z[i];

        if (delta_z > max_step_height_) {
            grid_msg.data[i] = 100; // Physical obstacle / step hazard
        } else if (min_intensity[i] < low_intensity_threshold) {
            grid_msg.data[i] = 80;  // False floor / low-reflection void
        } else {
            grid_msg.data[i] = 0;   // Free traversable space
        }
    }

    return grid_msg;
}

} // namespace lunar_slam