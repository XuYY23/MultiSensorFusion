//#pragma once
//
//#include "structs.h"
//
//// ������ȡ����
//class FeatureExtractor {
//public:
//    // ��ȡͼ������
//    ObjectFeature extractVisualFeatures(const cv::Mat& image, const cv::Rect2d& bbox);
//
//    // ��ȡ�״�����
//    ObjectFeature extractRadarFeatures(const std::vector<double>& radar_data);
//
//    // ��ȡ��Ƶ����
//    ObjectFeature extractAudioFeatures(const std::vector<double>& audio_data);
//
//    // ����������������֮������ƶ�
//    double calculateFeatureSimilarity(const ObjectFeature& f1, const ObjectFeature& f2);
//};