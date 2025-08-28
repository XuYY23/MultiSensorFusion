#include "SpatioTemporalAligner.h"

SpatioTemporalAligner::SpatioTemporalAligner(const std::map<std::string, SensorCalibration>& calibrations) : 
    calibrations_(calibrations) {
}

// �Ե�����������пռ���루����������->ȫ�����꣩
void SpatioTemporalAligner::alignSpatial(Detection& detection) {
    auto it = calibrations_.find(detection.sensor_id);
    if (it == calibrations_.end()) {
        // ���û��У׼��Ϣ�������Ѿ���ȫ������ϵ
        detection.position_global = detection.position;
        detection.velocity_global = detection.velocity;
        return;
    }

    const SensorCalibration& calib = it->second;

    // Ӧ����ת��ƽ�ƽ�����������ת����ȫ������
    detection.position_global = calib.rotation * detection.position + calib.translation;
    detection.velocity_global = calib.rotation * detection.velocity;
}

// �Ե������������ʱ����루������ʱ��->ϵͳʱ�䣩
void SpatioTemporalAligner::alignTemporal(Detection& detection) {
    auto it = calibrations_.find(detection.sensor_id);
    if (it == calibrations_.end()) {
        // ���û��У׼��Ϣ������ʱ���Ѿ�����
        return;
    }

    const SensorCalibration& calib = it->second;

    // ����Ӵ��������������ڵ�ʱ�䣨�룩
    auto now = std::chrono::system_clock::now();
    auto sensor_uptime = std::chrono::duration_cast<std::chrono::seconds>(now - detection.timestamp);

    // ����ʱ��Ư�ƣ��룩
    double drift = sensor_uptime.count() * calib.time_drift / 1e6;

    // Ӧ��ʱ��ƫ�ƺ�Ư��У��
    detection.timestamp += calib.time_offset + std::chrono::microseconds((int)(drift * 1e6));
}

// ��һ����������ʱ��ͬ���������ǲ�ֵ��Ŀ��ʱ���
std::vector<Detection> SpatioTemporalAligner::synchronizeDetections(const std::vector<Detection>& detections, Timestamp target_time) {
    // ������������
    std::map<std::string, std::vector<Detection>> sensor_detections;
    for (const auto& det : detections) {
        sensor_detections[det.sensor_id].push_back(det);
    }

    std::vector<Detection> synchronized;

    // ��ÿ�������������ݽ���ʱ��ͬ��
    for (auto& [sensor_id, dets] : sensor_detections) {
        // ����ô������ļ����
        std::sort(dets.begin(), dets.end(),
            [](const Detection& a, const Detection& b) {
                return a.timestamp < b.timestamp;
            }
        );

        // ����Ŀ��ʱ�丽�������������
        auto it = std::upper_bound(dets.begin(), dets.end(), target_time,
            [](Timestamp t, const Detection& d) {
                return t < d.timestamp;
            }
        );

        if (it != dets.end() && it != dets.begin()) {
            // ��ǰ���������ݣ����в�ֵ
            Detection prev = *(it - 1);
            Detection curr = *it;
            Detection interpolated = interpolateDetection(prev, curr, target_time);
            synchronized.push_back(interpolated);
        } else if (it != dets.end()) {
            // ֻ��ǰ�����ݣ�ֱ��ʹ��
            Detection det = *it;
            alignSpatial(det);
            synchronized.push_back(det);
        } else if (!dets.empty()) {
            // ֻ�к������ݣ�ֱ��ʹ��
            Detection det = dets.back();
            alignSpatial(det);
            synchronized.push_back(det);
        }
    }

    return synchronized;
}

// ��������ʱ���֮��ļ������ֵ������������ֵ
Detection SpatioTemporalAligner::interpolateDetection(const Detection& earlier, const Detection& later, Timestamp target_time) {

    // ����ʱ������ [0, 1]
    auto earlier_to_target = std::chrono::duration_cast<std::chrono::microseconds>(target_time - earlier.timestamp);
    auto earlier_to_later = std::chrono::duration_cast<std::chrono::microseconds>(later.timestamp - earlier.timestamp);

    if (earlier_to_later.count() == 0) {
        return earlier;  // ʱ����ͬ��ֱ�ӷ��ؽ���ļ����
    }

    double alpha = static_cast<double>(earlier_to_target.count()) / earlier_to_later.count();
    if (alpha < 0) {
        alpha = 0;
    }
    if (alpha > 1) {
        alpha = 1;
    }

    Detection interpolated;
    interpolated.sensor_id = earlier.sensor_id;
    interpolated.sensor_type = earlier.sensor_type;
    interpolated.timestamp = target_time;
    interpolated.local_id = earlier.local_id;  // ʹ�ý����ID
    interpolated.detected_class = earlier.detected_class;  // ��𲻲�ֵ��ʹ�ý����

    // ��ֵ���Ŷ�
    interpolated.class_confidence = earlier.class_confidence * (1 - alpha) + later.class_confidence * alpha;
    interpolated.detection_confidence = earlier.detection_confidence * (1 - alpha) + later.detection_confidence * alpha;

    // ��ֵλ�ú��ٶȣ�����������ϵ��
    interpolated.position = earlier.position * (1 - alpha) + later.position * alpha;
    interpolated.velocity = earlier.velocity * (1 - alpha) + later.velocity * alpha;

    // ��ֵ�߽������У�
    if (earlier.bbox.area() > 0 && later.bbox.area() > 0) {
        interpolated.bbox.x = earlier.bbox.x * (1 - alpha) + later.bbox.x * alpha;
        interpolated.bbox.y = earlier.bbox.y * (1 - alpha) + later.bbox.y * alpha;
        interpolated.bbox.width = earlier.bbox.width * (1 - alpha) + later.bbox.width * alpha;
        interpolated.bbox.height = earlier.bbox.height * (1 - alpha) + later.bbox.height * alpha;
    }

    // ��ֵ��������
    interpolated.features = interpolateFeatures(earlier.features, later.features, alpha);

    // ����ȫ������
    alignSpatial(interpolated);

    return interpolated;
}

// ����������ֵʵ��
FeatureVector SpatioTemporalAligner::interpolateFeatures(const FeatureVector& earlier, const FeatureVector& later, double alpha) {

    FeatureVector interpolated;

    // ��ֵ�Ӿ�����
    interpolated.visual_features = interpolateVector(earlier.visual_features, later.visual_features, alpha);

    // ��ֵ�״�����
    interpolated.radar_features = interpolateVector(earlier.radar_features, later.radar_features, alpha);

    // ��ֵ��Ƶ����
    interpolated.audio_features = interpolateVector(earlier.audio_features, later.audio_features, alpha);

    // ��ֵ��״����
    interpolated.shape_features = interpolateVector(earlier.shape_features, later.shape_features, alpha);

    // ��ֵ�˶�����
    interpolated.motion_features = interpolateVector(earlier.motion_features, later.motion_features, alpha);

    return interpolated;
}

// ������ֵ��������
std::vector<double> SpatioTemporalAligner::interpolateVector(const std::vector<double>& earlier, const std::vector<double>& later, double alpha) {

    std::vector<double> interpolated;

    // �������������Ϊ�գ����ؿ�����
    if (earlier.empty() && later.empty()) {
        return interpolated;
    }

    // ���һ��Ϊ�գ�������һ��
    if (earlier.empty()) {
        return later;
    }
    if (later.empty()) {
        return earlier;
    }

    // ȡ��С���Ƚ��в�ֵ
    size_t min_size = std::min(earlier.size(), later.size());
    interpolated.resize(min_size);

    // ���Բ�ֵ
    for (size_t i = 0; i < min_size; ++i) {
        interpolated[i] = earlier[i] * (1 - alpha) + later[i] * alpha;
    }

    return interpolated;
}