#include "lunar_slam/thermal_processing.hpp"
#include <filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

namespace lunar_slam {

cv::Mat ThermalProcessor::normalize16To8Bit(const cv::Mat& raw_16bit) {
    if (raw_16bit.empty()) return cv::Mat();

    // Handle both 16-bit raw and already 8-bit inputs
    if (raw_16bit.type() == CV_8UC1) {
        return raw_16bit.clone();
    }

    double min_val, max_val;
    cv::minMaxLoc(raw_16bit, &min_val, &max_val);

    cv::Mat normalized_8bit;
    if (max_val - min_val > 0) {
        double alpha = 255.0 / (max_val - min_val);
        double beta = -min_val * alpha;
        raw_16bit.convertTo(normalized_8bit, CV_8UC1, alpha, beta);
    } else {
        raw_16bit.convertTo(normalized_8bit, CV_8UC1, 0.0);
    }

    return normalized_8bit;
}

cv::Mat ThermalProcessor::enhanceThermalFrame(const cv::Mat& frame_8bit) {
    if (frame_8bit.empty()) return cv::Mat();

    // 1. Bilateral filter to reduce thermal noise while preserving sharp crater/rock edges
    cv::Mat denoised;
    cv::bilateralFilter(frame_8bit, denoised, 5, 75.0, 75.0);

    // 2. Apply CLAHE for local contrast enhancement in low-gradient thermal areas
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    cv::Mat enhanced;
    clahe->apply(denoised, enhanced);

    return enhanced;
}

ThermalFeatureResult ThermalProcessor::extractAndMatchFeatures(const cv::Mat& prev_frame, const cv::Mat& curr_frame) {
    ThermalFeatureResult result;
    if (prev_frame.empty() || curr_frame.empty()) return result;

    // Detect ORB keypoints & compute descriptors
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
    cv::Mat desc_prev, desc_curr;

    orb->detectAndCompute(prev_frame, cv::noArray(), result.keypoints_prev, desc_prev);
    orb->detectAndCompute(curr_frame, cv::noArray(), result.keypoints_curr, desc_curr);

    if (desc_prev.empty() || desc_curr.empty()) return result;

    // Match descriptors using Hamming distance
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> raw_matches;
    matcher.match(desc_prev, desc_curr, raw_matches);

    // Filter top matches based on distance threshold
    std::sort(raw_matches.begin(), raw_matches.end());
    size_t num_good = std::min(raw_matches.size(), static_cast<size_t>(100));
    result.matches.assign(raw_matches.begin(), raw_matches.begin() + num_good);

    // Draw matches for visualization/debugging
    cv::drawMatches(prev_frame, result.keypoints_prev, 
                    curr_frame, result.keypoints_curr, 
                    result.matches, result.matched_image, 
                    cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255));

    return result;
}

std::vector<std::string> loadDatasetSequence(const std::string& folder_path) {
    std::vector<std::string> filepaths;
    if (!fs::exists(folder_path)) {
        return filepaths;
    }

    for (const auto& entry : fs::directory_iterator(folder_path)) {
        std::string ext = entry.path().extension().string();
        if (ext == ".png" || ext == ".tiff" || ext == ".jpg" || ext == ".bmp") {
            filepaths.push_back(entry.path().string());
        }
    }

    std::sort(filepaths.begin(), filepaths.end());
    return filepaths;
}

} // namespace lunar_slam