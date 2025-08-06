//#include "FeatureExtractor.h"
//
//// ������ȡ��ʵ��
//ObjectFeature FeatureExtractor::extractVisualFeatures(const cv::Mat& image, const cv::Rect2d& bbox) {
//    ObjectFeature feature;
//
//    // ��ȡROI
//    cv::Rect roi(
//        static_cast<int>(bbox.x),
//        static_cast<int>(bbox.y),
//        static_cast<int>(bbox.width),
//        static_cast<int>(bbox.height)
//    );
//    roi &= cv::Rect(0, 0, image.cols, image.rows);  // ȷ��ROI��ͼ��Χ��
//
//    if (roi.width <= 0 || roi.height <= 0) {
//        return feature;
//    }
//
//    cv::Mat roi_img = image(roi);
//
//    // ��ʵ�֣���ȡ��ɫֱ��ͼ��Ϊ�Ӿ�����
//    cv::Mat hsv;
//    cv::cvtColor(roi_img, hsv, cv::COLOR_BGR2HSV);
//
//    // ����H��S��V����ͨ����ֱ��ͼ
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
//    // ��ֱ��ͼ���ݴ�����������
//    feature.visual_features.reserve(h_bins * s_bins * v_bins);
//    for (int h = 0; h < h_bins; ++h) {
//        for (int s = 0; s < s_bins; ++s) {
//            for (int v = 0; v < v_bins; ++v) {
//                feature.visual_features.push_back(hist.at<float>(h, s, v));
//            }
//        }
//    }
//
//    // ��ȡ��״�������򻯣���߱ȵȣ�
//    feature.shape_descriptor.push_back(bbox.width / bbox.height);  // ��߱�
//    feature.shape_descriptor.push_back(bbox.area());               // ���
//
//    return feature;
//}
//
//// ��ȡ�״�����
//ObjectFeature FeatureExtractor::extractRadarFeatures(const std::vector<double>& radar_data) {
//    ObjectFeature feature;
//
//    // ��ʵ�֣������״����ݰ������롢�ٶȡ���λ�Ǻ�RCS
//    if (radar_data.size() >= 4) {
//        // ����
//        feature.radar_signature.push_back(radar_data[0]);
//        // �ٶ�
//        feature.radar_signature.push_back(radar_data[1]);
//        // ��λ��
//        feature.radar_signature.push_back(radar_data[2]);
//        // RCS (�״�����)
//        feature.radar_signature.push_back(radar_data[3]);
//
//        // ���ٶ���Ϊ�˶�����
//        feature.motion_features.push_back(radar_data[1]);
//    }
//
//    return feature;
//}
//
//// ��ȡ��Ƶ����
//ObjectFeature FeatureExtractor::extractAudioFeatures(const std::vector<double>& audio_data) {
//    ObjectFeature feature;
//
//    // ��ʵ�֣���ȡ��Ƶ�Ļ�������
//    if (audio_data.empty()) {
//        return feature;
//    }
//
//    // ��������
//    double energy = 0;
//    for (double sample : audio_data) {
//        energy += sample * sample;
//    }
//    energy /= audio_data.size();
//    feature.audio_features.push_back(energy);
//
//    // ��Ƶ���������򻯣�
//    feature.audio_features.push_back(audio_data.size());  // ��Ƶ����
//
//    return feature;
//}
//
//// ����������������֮������ƶ�
//double FeatureExtractor::calculateFeatureSimilarity(const ObjectFeature& f1, const ObjectFeature& f2) {
//    double similarity = 0.0;
//    int feature_count = 0;
//
//    // ������״�������ƶ�
//    if (!f1.shape_descriptor.empty() && !f2.shape_descriptor.empty()) {
//        double sum = 0.0;
//        size_t min_size = std::min(f1.shape_descriptor.size(), f2.shape_descriptor.size());
//        for (size_t i = 0; i < min_size; ++i) {
//            // ʹ�ù�һ��ŷ�Ͼ���������ƶ�
//            double diff = f1.shape_descriptor[i] - f2.shape_descriptor[i];
//            sum += 1.0 / (1.0 + std::abs(diff));  // ת��Ϊ���ƶ� (0-1)
//        }
//        similarity += sum / min_size;
//        feature_count++;
//    }
//
//    // �����Ӿ��������ƶ�
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
//    // �����״��������ƶ�
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
//    // ������Ƶ�������ƶ�
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
//    // �����˶��������ƶ�
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
//    // ƽ���������������ƶ�
//    return feature_count > 0 ? similarity / feature_count : 0.0;
//}