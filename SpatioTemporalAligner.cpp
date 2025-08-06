#include "SpatioTemporalAligner.h"

SpatioTemporalAligner::SpatioTemporalAligner(const std::map<std::string, SensorCalibration>& calibrations) : 
    calibrations_(calibrations) {
}

// 对单个检测结果进行空间对齐（传感器坐标->全局坐标）
void SpatioTemporalAligner::alignSpatial(Detection& detection) {
    auto it = calibrations_.find(detection.sensor_id);
    if (it == calibrations_.end()) {
        // 如果没有校准信息，假设已经在全局坐标系
        detection.position_global = detection.position;
        detection.velocity_global = detection.velocity;
        return;
    }

    const SensorCalibration& calib = it->second;

    // 应用旋转和平移将传感器坐标转换到全局坐标
    detection.position_global = calib.rotation * detection.position + calib.translation;
    detection.velocity_global = calib.rotation * detection.velocity;
}

// 对单个检测结果进行时间对齐（传感器时间->系统时间）
void SpatioTemporalAligner::alignTemporal(Detection& detection) {
    auto it = calibrations_.find(detection.sensor_id);
    if (it == calibrations_.end()) {
        // 如果没有校准信息，假设时间已经对齐
        return;
    }

    const SensorCalibration& calib = it->second;

    // 计算从传感器启动到现在的时间（秒）
    auto now = std::chrono::system_clock::now();
    auto sensor_uptime = std::chrono::duration_cast<std::chrono::seconds>(now - detection.timestamp);

    // 计算时间漂移（秒）
    double drift = sensor_uptime.count() * calib.time_drift / 1e6;

    // 应用时间偏移和漂移校正
    detection.timestamp += calib.time_offset + std::chrono::microseconds((int)(drift * 1e6));
}

// 对一组检测结果进行时间同步，将它们插值到目标时间点
std::vector<Detection> SpatioTemporalAligner::synchronizeDetections(const std::vector<Detection>& detections, Timestamp target_time) {
    // 按传感器分组
    std::map<std::string, std::vector<Detection>> sensor_detections;
    for (const auto& det : detections) {
        sensor_detections[det.sensor_id].push_back(det);
    }

    std::vector<Detection> synchronized;

    // 对每个传感器的数据进行时间同步
    for (auto& [sensor_id, dets] : sensor_detections) {
        // 排序该传感器的检测结果
        std::sort(dets.begin(), dets.end(),
            [](const Detection& a, const Detection& b) {
                return a.timestamp < b.timestamp;
            }
        );

        // 查找目标时间附近的两个检测结果
        auto it = std::upper_bound(dets.begin(), dets.end(), target_time,
            [](Timestamp t, const Detection& d) {
                return t < d.timestamp;
            }
        );

        if (it != dets.end() && it != dets.begin()) {
            // 有前后两个数据，进行插值
            Detection prev = *(it - 1);
            Detection curr = *it;
            Detection interpolated = interpolateDetection(prev, curr, target_time);
            synchronized.push_back(interpolated);
        } else if (it != dets.end()) {
            // 只有前向数据，直接使用
            Detection det = *it;
            alignSpatial(det);
            synchronized.push_back(det);
        } else if (!dets.empty()) {
            // 只有后向数据，直接使用
            Detection det = dets.back();
            alignSpatial(det);
            synchronized.push_back(det);
        }
    }

    return synchronized;
}

// 计算两个时间点之间的检测结果插值，包括特征插值
Detection SpatioTemporalAligner::interpolateDetection(const Detection& earlier, const Detection& later, Timestamp target_time) {

    // 计算时间差比例 [0, 1]
    auto earlier_to_target = std::chrono::duration_cast<std::chrono::microseconds>(target_time - earlier.timestamp);
    auto earlier_to_later = std::chrono::duration_cast<std::chrono::microseconds>(later.timestamp - earlier.timestamp);

    if (earlier_to_later.count() == 0) {
        return earlier;  // 时间相同，直接返回较早的检测结果
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
    interpolated.local_id = earlier.local_id;  // 使用较早的ID
    interpolated.detected_class = earlier.detected_class;  // 类别不插值，使用较早的

    // 插值置信度
    interpolated.class_confidence = earlier.class_confidence * (1 - alpha) + later.class_confidence * alpha;
    interpolated.detection_confidence = earlier.detection_confidence * (1 - alpha) + later.detection_confidence * alpha;

    // 插值位置和速度（传感器坐标系）
    interpolated.position = earlier.position * (1 - alpha) + later.position * alpha;
    interpolated.velocity = earlier.velocity * (1 - alpha) + later.velocity * alpha;

    // 插值边界框（如果有）
    if (earlier.bbox.area() > 0 && later.bbox.area() > 0) {
        interpolated.bbox.x = earlier.bbox.x * (1 - alpha) + later.bbox.x * alpha;
        interpolated.bbox.y = earlier.bbox.y * (1 - alpha) + later.bbox.y * alpha;
        interpolated.bbox.width = earlier.bbox.width * (1 - alpha) + later.bbox.width * alpha;
        interpolated.bbox.height = earlier.bbox.height * (1 - alpha) + later.bbox.height * alpha;
    }

    // 插值特征向量
    interpolated.features = interpolateFeatures(earlier.features, later.features, alpha);

    // 计算全局坐标
    alignSpatial(interpolated);

    return interpolated;
}

// 特征向量插值实现
FeatureVector SpatioTemporalAligner::interpolateFeatures(const FeatureVector& earlier, const FeatureVector& later, double alpha) {

    FeatureVector interpolated;

    // 插值视觉特征
    interpolated.visual_features = interpolateVector(earlier.visual_features, later.visual_features, alpha);

    // 插值雷达特征
    interpolated.radar_features = interpolateVector(earlier.radar_features, later.radar_features, alpha);

    // 插值音频特征
    interpolated.audio_features = interpolateVector(earlier.audio_features, later.audio_features, alpha);

    // 插值形状特征
    interpolated.shape_features = interpolateVector(earlier.shape_features, later.shape_features, alpha);

    // 插值运动特征
    interpolated.motion_features = interpolateVector(earlier.motion_features, later.motion_features, alpha);

    return interpolated;
}

// 向量插值辅助函数
std::vector<double> SpatioTemporalAligner::interpolateVector(const std::vector<double>& earlier, const std::vector<double>& later, double alpha) {

    std::vector<double> interpolated;

    // 如果两个向量都为空，返回空向量
    if (earlier.empty() && later.empty()) {
        return interpolated;
    }

    // 如果一个为空，返回另一个
    if (earlier.empty()) {
        return later;
    }
    if (later.empty()) {
        return earlier;
    }

    // 取最小长度进行插值
    size_t min_size = std::min(earlier.size(), later.size());
    interpolated.resize(min_size);

    // 线性插值
    for (size_t i = 0; i < min_size; ++i) {
        interpolated[i] = earlier[i] * (1 - alpha) + later[i] * alpha;
    }

    return interpolated;
}