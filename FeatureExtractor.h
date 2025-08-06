//#pragma once
//
//#include "structs.h"
//
//// 特征提取器类
//class FeatureExtractor {
//public:
//    // 提取图像特征
//    ObjectFeature extractVisualFeatures(const cv::Mat& image, const cv::Rect2d& bbox);
//
//    // 提取雷达特征
//    ObjectFeature extractRadarFeatures(const std::vector<double>& radar_data);
//
//    // 提取音频特征
//    ObjectFeature extractAudioFeatures(const std::vector<double>& audio_data);
//
//    // 计算两个特征向量之间的相似度
//    double calculateFeatureSimilarity(const ObjectFeature& f1, const ObjectFeature& f2);
//};