#ifndef LUNAR_SLAM_TRAVERSABILITY_MAPPER_HPP_
#define LUNAR_SLAM_TRAVERSABILITY_MAPPER_HPP_

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

namespace lunar_slam {

class TraversabilityMapper {
public:
    TraversabilityMapper(double resolution = 0.1, int grid_size = 200, double max_step_height = 0.15);

    nav_msgs::msg::OccupancyGrid generateCostmap(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg);

private:
    double resolution_;
    int grid_size_;
    double max_step_height_;
    double origin_offset_;
};

} // namespace lunar_slam

#endif