//#include "FeatureExtractor.h"
//
//// 特征提取器实现
//ObjectFeature FeatureExtractor::extractVisualFeatures(const cv::Mat& image, const cv::Rect2d& bbox) {
//    ObjectFeature feature;
//
//    // 提取ROI
//    cv::Rect roi(
//        static_cast<int>(bbox.x),
//        static_cast<int>(bbox.y),
//        static_cast<int>(bbox.width),
//        static_cast<int>(bbox.height)
//    );
//    roi &= cv::Rect(0, 0, image.cols, image.rows);  // 确保ROI在图像范围内
//
//    if (roi.width <= 0 || roi.height <= 0) {
//        return feature;
//    }
//
//    cv::Mat roi_img = image(roi);
//
//    // 简化实现：提取颜色直方图作为视觉特征
//    cv::Mat hsv;
//    cv::cvtColor(roi_img, hsv, cv::COLOR_BGR2HSV);
//
//    // 计算H、S、V三个通道的直方图
//    int h_bins = 8, s_bins = 8, v_bins = 8;
//    int histSize[] = { h_bins, s_bins, v_bins };
//
//    float h_ranges[] = { 0, 180 };
//    float s_ranges[] = { 0, 256 };
//    float v_ranges[] = { 0, 256 };
//    const float* ranges[] = { h_ranges, s_ranges, v_ranges };
//
//    int channels[] = { 0, 1, 2 };
//
//    cv::Mat hist;
//    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 3, histSize, ranges, true, false);
//    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
//
//    // 将直方图数据存入特征向量
//    feature.visual_features.reserve(h_bins * s_bins * v_bins);
//    for (int h = 0; h < h_bins; ++h) {
//        for (int s = 0; s < s_bins; ++s) {
//            for (int v = 0; v < v_bins; ++v) {
//                feature.visual_features.push_back(hist.at<float>(h, s, v));
//            }
//        }
//    }
//
//    // 提取形状特征（简化：宽高比等）
//    feature.shape_descriptor.push_back(bbox.width / bbox.height);  // 宽高比
//    feature.shape_descriptor.push_back(bbox.area());               // 面积
//
//    return feature;
//}
//
//// 提取雷达特征
//ObjectFeature FeatureExtractor::extractRadarFeatures(const std::vector<double>& radar_data) {
//    ObjectFeature feature;
//
//    // 简化实现：假设雷达数据包含距离、速度、方位角和RCS
//    if (radar_data.size() >= 4) {
//        // 距离
//        feature.radar_signature.push_back(radar_data[0]);
//        // 速度
//        feature.radar_signature.push_back(radar_data[1]);
//        // 方位角
//        feature.radar_signature.push_back(radar_data[2]);
//        // RCS (雷达截面积)
//        feature.radar_signature.push_back(radar_data[3]);
//
//        // 将速度作为运动特征
//        feature.motion_features.push_back(radar_data[1]);
//    }
//
//    return feature;
//}
//
//// 提取音频特征
//ObjectFeature FeatureExtractor::extractAudioFeatures(const std::vector<double>& audio_data) {
//    ObjectFeature feature;
//
//    // 简化实现：提取音频的基本特征
//    if (audio_data.empty()) {
//        return feature;
//    }
//
//    // 计算能量
//    double energy = 0;
//    for (double sample : audio_data) {
//        energy += sample * sample;
//    }
//    energy /= audio_data.size();
//    feature.audio_features.push_back(energy);
//
//    // 简单频谱特征（简化）
//    feature.audio_features.push_back(audio_data.size());  // 音频长度
//
//    return feature;
//}
//
//// 计算两个特征向量之间的相似度
//double FeatureExtractor::calculateFeatureSimilarity(const ObjectFeature& f1, const ObjectFeature& f2) {
//    double similarity = 0.0;
//    int feature_count = 0;
//
//    // 计算形状特征相似度
//    if (!f1.shape_descriptor.empty() && !f2.shape_descriptor.empty()) {
//        double sum = 0.0;
//        size_t min_size = std::min(f1.shape_descriptor.size(), f2.shape_descriptor.size());
//        for (size_t i = 0; i < min_size; ++i) {
//            // 使用归一化欧氏距离计算相似度
//            double diff = f1.shape_descriptor[i] - f2.shape_descriptor[i];
//            sum += 1.0 / (1.0 + std::abs(diff));  // 转换为相似度 (0-1)
//        }
//        similarity += sum / min_size;
//        feature_count++;
//    }
//
//    // 计算视觉特征相似度
//    if (!f1.visual_features.empty() && !f2.visual_features.empty()) {
//        double sum = 0.0;
//        size_t min_size = std::min(f1.visual_features.size(), f2.visual_features.size());
//        for (size_t i = 0; i < min_size; ++i) {
//            double diff = f1.visual_features[i] - f2.visual_features[i];
//            sum += 1.0 / (1.0 + std::abs(diff));
//        }
//        similarity += sum / min_size;
//        feature_count++;
//    }
//
//    // 计算雷达特征相似度
//    if (!f1.radar_signature.empty() && !f2.radar_signature.empty()) {
//        double sum = 0.0;
//        size_t min_size = std::min(f1.radar_signature.size(), f2.radar_signature.size());
//        for (size_t i = 0; i < min_size; ++i) {
//            double diff = f1.radar_signature[i] - f2.radar_signature[i];
//            sum += 1.0 / (1.0 + std::abs(diff));
//        }
//        similarity += sum / min_size;
//        feature_count++;
//    }
//
//    // 计算音频特征相似度
//    if (!f1.audio_features.empty() && !f2.audio_features.empty()) {
//        double sum = 0.0;
//        size_t min_size = std::min(f1.audio_features.size(), f2.audio_features.size());
//        for (size_t i = 0; i < min_size; ++i) {
//            double diff = f1.audio_features[i] - f2.audio_features[i];
//            sum += 1.0 / (1.0 + std::abs(diff));
//        }
//        similarity += sum / min_size;
//        feature_count++;
//    }
//
//    // 计算运动特征相似度
//    if (!f1.motion_features.empty() && !f2.motion_features.empty()) {
//        double sum = 0.0;
//        size_t min_size = std::min(f1.motion_features.size(), f2.motion_features.size());
//        for (size_t i = 0; i < min_size; ++i) {
//            double diff = f1.motion_features[i] - f2.motion_features[i];
//            sum += 1.0 / (1.0 + std::abs(diff));
//        }
//        similarity += sum / min_size;
//        feature_count++;
//    }
//
//    // 平均所有特征的相似度
//    return feature_count > 0 ? similarity / feature_count : 0.0;
//}