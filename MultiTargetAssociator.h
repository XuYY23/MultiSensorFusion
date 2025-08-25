#pragma once

#include "ConstantVelocityModel.h"

// ��Ŀ������������𽫼����������Ŀ�����
class MultiTargetAssociator {
public:
    MultiTargetAssociator();
    ~MultiTargetAssociator();

    // �����¼����������Ŀ�꣨Ŀ����٣�
    std::vector<FusedObject> associateTargets(const std::vector<Detection>& new_detections,
                                              const std::vector<FusedObject>& existing_targets,
                                              Timestamp current_time);

    // �����¼����������Ŀ�꣨�޸��٣�
    std::vector<FusedObject> associateTargets(const std::vector<Detection>& detections, Timestamp current_time);

private:
    int next_global_id_;  // ��һ��ȫ��Ŀ��ID
    ConstantVelocityModel motion_model_;  // �˶�ģ��
};