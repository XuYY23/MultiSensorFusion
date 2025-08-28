#pragma once

#include "structs.h"

// �����ں����������������;��߼��ں�
class DataFusion {
public:
    // �������ںϣ��ں϶�������������
    FeatureVector fuseFeatures(const std::vector<Detection>& detections);

    // ���߼��ںϣ��ں϶�������������жϣ�����D-S֤�����ۣ�
    std::map<ObjectClass, double> fuseDecisions(const std::vector<Detection>& detections, double conf_threshold, bool& has_category_conflict);

    // λ���ںϣ��ں϶���������λ����Ϣ
    Eigen::Vector3d fusePositions(const std::vector<Detection>& detections, Eigen::Matrix3d& covariance);

    // �ٶ��ںϣ��ں϶����������ٶ���Ϣ
    Eigen::Vector3d fuseVelocities(const std::vector<Detection>& detections, Eigen::Matrix3d& covariance);
};
