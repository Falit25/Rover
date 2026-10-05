#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace lunar_slam {

struct ThermalFeatureResult {
    std::vector<cv::KeyPoint> keypoints_prev;
    std::vector<cv::KeyPoint> keypoints_curr;
    std::vector<cv::DMatch> matches;
    cv::Mat matched_image;
};

class ThermalProcessor {
public:
    ThermalProcessor() = default;

    cv::Mat normalize16To8Bit(const cv::Mat& raw_16bit);

    /**
     * @brief Enhances thermal contrast using CLAHE and reduces thermal sensor noise.
     */
    cv::Mat enhanceThermalFrame(const cv::Mat& frame_8bit);

    /**
     * @brief Extracts ORB keypoints and matches features between two consecutive thermal frames.
     */
    ThermalFeatureResult extractAndMatchFeatures(const cv::Mat& prev_frame, const cv::Mat& curr_frame);
};

/**
 * @brief Helper to load all frame paths from a sequence directory.
 */
std::vector<std::string> loadDatasetSequence(const std::string& folder_path);

} // namespace lunar_slam