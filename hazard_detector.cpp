#include "lunar_slam/hazard_detector.hpp"
#include "lunar_slam/mobility_risk_model.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <algorithm>
#include <cmath>
#include <queue>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace lunar_slam {

HazardDetector::HazardDetector(const HazardConfig& config) : config_(config) {
    origin_offset_ = (config_.grid_size * config_.resolution) / 2.0;
    hazard_grid_.resize(config_.grid_size * config_.grid_size);
}

nav_msgs::msg::OccupancyGrid HazardDetector::detectHazards(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg,
                                                           const cv::Mat* thermal_image,
                                                           const sensor_msgs::msg::CameraInfo* camera_info,
                                                           const geometry_msgs::msg::TransformStamped* lidar_to_thermal) {
    initializeGrid();

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    processPointCloud(cloud);
    computeElevationAndSlope();
    computeRoughnessAndStepHeight();  // NEW
    classifyHazards(thermal_image, camera_info, lidar_to_thermal);
    applyMorphologicalFiltering();

    nav_msgs::msg::OccupancyGrid grid_msg;
    grid_msg.header = cloud_msg->header;
    grid_msg.header.frame_id = "odom";
    grid_msg.info.resolution = config_.resolution;
    grid_msg.info.width = config_.grid_size;
    grid_msg.info.height = config_.grid_size;
    grid_msg.info.origin.position.x = -origin_offset_;
    grid_msg.info.origin.position.y = -origin_offset_;
    grid_msg.info.origin.position.z = 0.0;
    grid_msg.info.origin.orientation.w = 1.0;

    grid_msg.data.resize(config_.grid_size * config_.grid_size, -1);

    for (int y = 0; y < config_.grid_size; ++y) {
        for (int x = 0; x < config_.grid_size; ++x) {
            int idx = gridIndex(x, y);
            grid_msg.data[idx] = hazardToCost(hazard_grid_[idx].type);
        }
    }

    return grid_msg;
}

void HazardDetector::initializeGrid() {
    std::fill(hazard_grid_.begin(), hazard_grid_.end(), HazardCell());
    for (auto& cell : hazard_grid_) {
        cell.type = HazardType::UNKNOWN;
        cell.elevation = -1e9;
        cell.confidence = 0.0;
    }
}

void HazardDetector::processPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    std::vector<double> min_z(config_.grid_size * config_.grid_size, 1e9);
    std::vector<double> max_z(config_.grid_size * config_.grid_size, -1e9);
    std::vector<double> sum_z(config_.grid_size * config_.grid_size, 0.0);
    std::vector<int> count(config_.grid_size * config_.grid_size, 0);
    std::vector<double> min_intensity(config_.grid_size * config_.grid_size, 1e9);

    for (const auto& point : cloud->points) {
        int gx = static_cast<int>((point.x + origin_offset_) / config_.resolution);
        int gy = static_cast<int>((point.y + origin_offset_) / config_.resolution);

        if (!inBounds(gx, gy)) continue;

        int idx = gridIndex(gx, gy);
        min_z[idx] = std::min(min_z[idx], static_cast<double>(point.z));
        max_z[idx] = std::max(max_z[idx], static_cast<double>(point.z));
        sum_z[idx] += point.z;
        count[idx]++;
        min_intensity[idx] = std::min(min_intensity[idx], static_cast<double>(point.intensity));
    }

    for (int i = 0; i < config_.grid_size * config_.grid_size; ++i) {
        if (count[i] > 0) {
            hazard_grid_[i].elevation = sum_z[i] / count[i];
            hazard_grid_[i].point_count = count[i];
            hazard_grid_[i].confidence = std::min(1.0, count[i] / 10.0);
        }
    }
}

void HazardDetector::computeElevationAndSlope() {
    int kSize = 3;
    int half = kSize / 2;

    for (int y = 0; y < config_.grid_size; ++y) {
        for (int x = 0; x < config_.grid_size; ++x) {
            int idx = gridIndex(x, y);
            if (hazard_grid_[idx].point_count == 0) continue;

            double dz_dx = 0.0, dz_dy = 0.0;
            int valid_neighbors = 0;

            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    if (kx == 0 && ky == 0) continue;
                    int nx = x + kx;
                    int ny = y + ky;
                    if (!inBounds(nx, ny)) continue;
                    int nidx = gridIndex(nx, ny);
                    if (hazard_grid_[nidx].point_count == 0) continue;

                    double dz = hazard_grid_[nidx].elevation - hazard_grid_[idx].elevation;
                    dz_dx += dz * kx;
                    dz_dy += dz * ky;
                    valid_neighbors++;
                }
            }

            if (valid_neighbors > 0) {
                double grad_mag = std::sqrt(dz_dx * dz_dx + dz_dy * dz_dy) / (config_.resolution * valid_neighbors);
                hazard_grid_[idx].slope = std::atan(grad_mag) * 180.0 / M_PI;

                double d2z_dx2 = 0.0, d2z_dy2 = 0.0;
                int curv_count = 0;
                for (int ky = -half; ky <= half; ++ky) {
                    for (int kx = -half; kx <= half; ++kx) {
                        if (kx == 0 && ky == 0) continue;
                        int nx = x + kx;
                        int ny = y + ky;
                        if (!inBounds(nx, ny)) continue;
                        int nidx = gridIndex(nx, ny);
                        if (hazard_grid_[nidx].point_count == 0) continue;

                        double dz = hazard_grid_[nidx].elevation - hazard_grid_[idx].elevation;
                        d2z_dx2 += dz * kx * kx;
                        d2z_dy2 += dz * ky * ky;
                        curv_count++;
                    }
                }
                if (curv_count > 0) {
                    hazard_grid_[idx].curvature = (d2z_dx2 + d2z_dy2) / (config_.resolution * config_.resolution * curv_count);
                }
            }
        }
    }
}

void HazardDetector::computeRoughnessAndStepHeight() {
    for (int y = 0; y < config_.grid_size; ++y) {
        for (int x = 0; x < config_.grid_size; ++x) {
            int idx = gridIndex(x, y);
            if (hazard_grid_[idx].point_count == 0) continue;

            // Collect elevations in 3x3 neighborhood
            std::vector<double> neighbor_elevations;
            double max_elev = -1e9;
            double min_elev = 1e9;

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    if (kx == 0 && ky == 0) continue;
                    int nx = x + kx;
                    int ny = y + ky;
                    if (!inBounds(nx, ny)) continue;
                    int nidx = gridIndex(nx, ny);
                    if (hazard_grid_[nidx].point_count == 0) continue;

                    double elev = hazard_grid_[nidx].elevation;
                    neighbor_elevations.push_back(elev);
                    max_elev = std::max(max_elev, elev);
                    min_elev = std::min(min_elev, elev);
                }
            }

            // Roughness = std dev of neighbor elevations
            if (!neighbor_elevations.empty()) {
                double mean = 0.0;
                for (double e : neighbor_elevations) mean += e;
                mean /= neighbor_elevations.size();

                double var = 0.0;
                for (double e : neighbor_elevations) var += (e - mean) * (e - mean);
                hazard_grid_[idx].roughness = std::sqrt(var / neighbor_elevations.size());
            }

            // Step height = max elevation difference to neighbors
            if (max_elev > -1e8 && min_elev < 1e8) {
                hazard_grid_[idx].step_height = max_elev - min_elev;
            }
        }
    }
}

double HazardDetector::projectToThermalPixel(const Eigen::Vector3d& point_odom,
                                             const sensor_msgs::msg::CameraInfo* camera_info,
                                             const geometry_msgs::msg::TransformStamped* odom_to_thermal,
                                             int& u, int& v) const {
    if (!camera_info || !odom_to_thermal) {
        return -1.0;
    }

    // Transform point from odom frame to thermal camera frame
    tf2::Transform tf;
    tf2::fromMsg(odom_to_thermal->transform, tf);
    tf2::Vector3 p_odom(point_odom.x(), point_odom.y(), point_odom.z());
    tf2::Vector3 p_thermal = tf * p_odom;

    // Camera projection (pinhole model)
    double fx = camera_info->k[0];
    double fy = camera_info->k[4];
    double cx = camera_info->k[2];
    double cy = camera_info->k[5];

    if (p_thermal.z() <= 0.01) {
        return -1.0; // Behind camera or too close
    }

    u = static_cast<int>(fx * p_thermal.x() / p_thermal.z() + cx);
    v = static_cast<int>(fy * p_thermal.y() / p_thermal.z() + cy);

    // Check image bounds
    if (u < 0 || u >= camera_info->width || v < 0 || v >= camera_info->height) {
        return -1.0;
    }

    return p_thermal.z(); // Return depth for potential use
}

void HazardDetector::classifyHazards(const cv::Mat* thermal_image,
                                     const sensor_msgs::msg::CameraInfo* camera_info,
                                     const geometry_msgs::msg::TransformStamped* odom_to_thermal) {
    double max_slope_rad = config_.max_slope_deg * M_PI / 180.0;

    for (int y = 0; y < config_.grid_size; ++y) {
        for (int x = 0; x < config_.grid_size; ++x) {
            int idx = gridIndex(x, y);
            HazardCell& cell = hazard_grid_[idx];

            if (cell.point_count == 0) {
                cell.type = HazardType::UNKNOWN;
                continue;
            }

            double delta_z = 0.0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int nx = x + kx, ny = y + ky;
                    if (!inBounds(nx, ny)) continue;
                    int nidx = gridIndex(nx, ny);
                    if (hazard_grid_[nidx].point_count > 0) {
                        delta_z = std::max(delta_z, std::abs(hazard_grid_[nidx].elevation - cell.elevation));
                    }
                }
            }

            if (delta_z > config_.max_step_height) {
                cell.type = HazardType::STEEP_SLOPE;
                continue;
            }

            if (cell.slope > config_.max_slope_deg) {
                cell.type = HazardType::STEEP_SLOPE;
                continue;
            }

            if (cell.curvature > config_.crater_rim_curvature && cell.elevation > 0) {
                cell.type = HazardType::CRATER_RIM;
                continue;
            }

            if (cell.elevation > config_.rock_height_thresh && cell.confidence > 0.5) {
                cell.type = HazardType::ROCK;
                continue;
            }

            // Proper thermal projection using camera model and TF
            if (thermal_image && !thermal_image->empty() && camera_info && odom_to_thermal) {
                // Convert grid cell center to world coordinates
                double wx = -origin_offset_ + (x + 0.5) * config_.resolution;
                double wy = -origin_offset_ + (y + 0.5) * config_.resolution;
                double wz = cell.elevation;
                
                Eigen::Vector3d point_odom(wx, wy, wz);
                int u, v;
                double depth = projectToThermalPixel(point_odom, camera_info, odom_to_thermal, u, v);
                
                if (depth > 0) {
                    double temp = thermal_image->at<uint16_t>(v, u) / 100.0; // Convert from centi-Kelvin
                    cell.temperature = temp;
                    if (temp < config_.cold_trap_temp_thresh) {
                        cell.type = HazardType::COLD_TRAP;
                        continue;
                    }
                }
            }

            cell.type = HazardType::FREE;
        }
    }
}

void HazardDetector::applyMorphologicalFiltering() {
    std::vector<HazardCell> filtered = hazard_grid_;
    cv::Mat hazard_mat(config_.grid_size, config_.grid_size, CV_8UC1);
    for (int y = 0; y < config_.grid_size; ++y) {
        for (int x = 0; x < config_.grid_size; ++x) {
            hazard_mat.at<uchar>(y, x) = static_cast<uchar>(hazard_grid_[gridIndex(x, y)].type);
        }
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(hazard_mat, hazard_mat, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(hazard_mat, hazard_mat, cv::MORPH_OPEN, kernel);

    for (int y = 0; y < config_.grid_size; ++y) {
        for (int x = 0; x < config_.grid_size; ++x) {
            hazard_grid_[gridIndex(x, y)].type = static_cast<HazardType>(hazard_mat.at<uchar>(y, x));
        }
    }
}

int8_t HazardDetector::hazardToCost(HazardType type) const {
    switch (type) {
        case HazardType::FREE: return 0;
        case HazardType::UNKNOWN: return -1;
        case HazardType::CRATER_RIM: return 90;
        case HazardType::ROCK: return 85;
        case HazardType::COLD_TRAP: return 95;
        case HazardType::STEEP_SLOPE: return 100;
        case HazardType::LOOSE_REGOLITH: return 70;
        default: return 50;
    }
}

std::vector<TerrainFeatures> HazardDetector::getTerrainFeatures() const {
    std::vector<TerrainFeatures> features(config_.grid_size * config_.grid_size);
    
    for (int y = 0; y < config_.grid_size; ++y) {
        for (int x = 0; x < config_.grid_size; ++x) {
            int idx = gridIndex(x, y);
            const HazardCell& cell = hazard_grid_[idx];
            TerrainFeatures& f = features[idx];
            
            if (cell.point_count == 0) {
                // Unknown cell - set defaults
                f.elevation = 0.0;
                f.slope_deg = 0.0;
                f.curvature = 0.0;
                f.roughness = 1.0;  // High roughness for unknown
                f.step_height = 1.0;
                f.temperature_k = 250.0;
                f.thermal_contrast = 0.0;
                f.thermal_gradient = 0.0;
                continue;
            }
            
            // Real terrain features from LiDAR
            f.elevation = cell.elevation;
            f.slope_deg = cell.slope;
            f.curvature = cell.curvature;
            f.roughness = cell.roughness;
            f.step_height = cell.step_height;
            f.temperature_k = cell.temperature;
            
            // Thermal contrast/gradient would need thermal image access
            // For now set defaults - can be enhanced later
            f.thermal_contrast = 0.0;
            f.thermal_gradient = 0.0;
        }
    }
    
    return features;
}

}  // namespace lunar_slam