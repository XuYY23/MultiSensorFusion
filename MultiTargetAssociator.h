#pragma once

#include "ConstantVelocityModel.h"

// 多目标关联器，负责将检测结果与已有目标关联
class MultiTargetAssociator {
public:
    MultiTargetAssociator();
    ~MultiTargetAssociator();

    // 关联新检测结果与已有目标（目标跟踪）
    std::vector<FusedObject> associateTargets(const std::vector<Detection>& new_detections,
                                              const std::vector<FusedObject>& existing_targets,
                                              Timestamp current_time);

    // 关联新检测结果与已有目标（无跟踪）
    std::vector<FusedObject> associateTargets(const std::vector<Detection>& detections, Timestamp current_time);

private:
    int next_global_id_;  // 下一个全局目标ID
    ConstantVelocityModel motion_model_;  // 运动模型
};