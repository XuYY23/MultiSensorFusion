#pragma once

#include "ConstantVelocityModel.h"

// ��Ŀ������������𽫼����������Ŀ�����
class MultiTargetAssociator {
public:
    MultiTargetAssociator();
    ~MultiTargetAssociator();

    // �����¼����������Ŀ��
    std::vector<FusedObject> associateTargets(const std::vector<Detection>& new_detections,
                                              const std::vector<FusedObject>& existing_targets,
                                              Timestamp current_time);

    std::vector<FusedObject> associateTargets(const std::vector<Detection>& detections, Timestamp current_time);

    // ������������������ƶ�
    double calculateSimilarity(const Detection& a, const Detection& b);

    // �����������������ƶ�
    double calculateFeatureSimilarity(const FeatureVector& a, const FeatureVector& b);

    // ���������������������ƶ�
    double cosineSimilarity(const Eigen::Vector3d& a, const Eigen::Vector3d& b);
    double cosineSimilarity(const std::vector<double>& a, const std::vector<double>& b);

	// �����˹���ƶȣ�����λ�ú��ٶȵ����Ͼ��룩
    double gaussianSimilarity(const double& dis, const double& scale);

private:
    // ʹ��dlib��ʵ�ֵ��������㷨
    std::vector<std::pair<int, int>> hungarianAlgorithm(const std::vector<std::vector<double>>& cost_matrix);

    int next_global_id_;  // ��һ��ȫ��Ŀ��ID
    ConstantVelocityModel motion_model_;  // �˶�ģ��
};