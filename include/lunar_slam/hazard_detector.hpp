#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <vector>
#include <string>
#include "lunar_slam/mobility_risk_model.hpp"

namespace lunar_slam {

enum class HazardType : uint8_t {
    UNKNOWN = 0,
    FREE = 1,
    CRATER_RIM = 2,
    ROCK = 3,
    COLD_TRAP = 4,
    STEEP_SLOPE = 5,
    LOOSE_REGOLITH = 6
};

struct HazardConfig {
    double resolution = 0.1;
    int grid_size = 200;
    double max_step_height = 0.15;
    double max_slope_deg = 20.0;
    double crater_rim_curvature = 0.5;
    double rock_height_thresh = 0.2;
    double cold_trap_temp_thresh = 100.0;
    double thermal_noise_sigma = 5.0;
};

struct HazardCell {
    HazardType type = HazardType::UNKNOWN;
    double elevation = 0.0;
    double slope = 0.0;
    double curvature = 0.0;
    double roughness = 0.0;       // NEW: standard deviation of elevation in neighborhood
    double step_height = 0.0;     // NEW: max elevation difference to neighbors
    double temperature = 0.0;
    double confidence = 0.0;
    int point_count = 0;
};



class HazardDetector {
public:
    explicit HazardDetector(const HazardConfig& config = HazardConfig());

    nav_msgs::msg::OccupancyGrid detectHazards(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg,
                                               const cv::Mat* thermal_image = nullptr,
                                               const sensor_msgs::msg::CameraInfo* camera_info = nullptr,
                                               const geometry_msgs::msg::TransformStamped* lidar_to_thermal = nullptr);
    
    // NEW: Get terrain features for mobility risk model
    std::vector<TerrainFeatures> getTerrainFeatures() const;
    const std::vector<HazardCell>& getHazardGrid() const { return hazard_grid_; }
    void setConfig(const HazardConfig& config) { config_ = config; }

private:
    HazardConfig config_;
    std::vector<HazardCell> hazard_grid_;
    double origin_offset_ = 0.0;

    void initializeGrid();
    void processPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);
    void computeElevationAndSlope();
    void computeRoughnessAndStepHeight();  // NEW
    void classifyHazards(const cv::Mat* thermal_image,
                         const sensor_msgs::msg::CameraInfo* camera_info,
                         const geometry_msgs::msg::TransformStamped* odom_to_thermal);
    void applyMorphologicalFiltering();
    double projectToThermalPixel(const Eigen::Vector3d& point_odom,
                                 const sensor_msgs::msg::CameraInfo* camera_info,
                                 const geometry_msgs::msg::TransformStamped* odom_to_thermal,
                                 int& u, int& v) const;
    int gridIndex(int x, int y) const { return y * config_.grid_size + x; }
    bool inBounds(int x, int y) const { return x >= 0 && x < config_.grid_size && y >= 0 && y < config_.grid_size; }
    int8_t hazardToCost(HazardType type) const;
};

}  // namespace lunar_slam