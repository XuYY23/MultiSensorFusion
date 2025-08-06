#pragma once

#include "SpatioTemporalAligner.h"
#include "FeatureExtractor.h"
#include "MultiTargetAssociator.h"
#include "DataFusion.h"
#include "JsonOutput.h"

// 多模态多传感器目标检测融合系统主类
class MultimodalDetectorSystem {
public:
    MultimodalDetectorSystem(const std::map<std::string, SensorCalibration>& calibrations);

    // 添加新的检测结果
    void addDetections(const std::vector<Detection>& detections);

    // 处理当前所有检测结果
    void processDetections();

    // 获取当前融合结果
    const std::vector<FusedObject>& getFusedObjects() const;

    // 保存融合结果到JSON文件
    bool saveResults(const std::string& filename);

    // 设置时间同步的目标时间（如果不设置则使用最新检测时间）
    void setTargetTimestamp(Timestamp target_time);

private:
    SpatioTemporalAligner aligner_;       // 时空对齐器
    MultiTargetAssociator associator_;    // 多目标关联器
    DataFusion fusion_;                   // 数据融合器
    JsonOutput json_output_;              // JSON输出器

    std::vector<Detection> current_detections_;  // 当前检测结果队列
    std::vector<FusedObject> fused_objects_;     // 融合后的目标

    Timestamp target_timestamp_;  // 时间同步的目标时间
    bool use_custom_target_time_; // 是否使用自定义目标时间
};