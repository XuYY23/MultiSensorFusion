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

    // ������������������ƶ�
    double calculateSimilarity(const Detection& a, const Detection& b);

    // �����������������ƶ�
    double calculateFeatureSimilarity(const FeatureVector& a, const FeatureVector& b);

    // ���������������������ƶ�
    double cosineSimilarity(const std::vector<double>& a, const std::vector<double>& b);

private:
    // ʹ��dlib��ʵ�ֵ��������㷨
    std::vector<std::pair<int, int>> hungarianAlgorithm(const std::vector<std::vector<double>>& cost_matrix);

    int next_global_id_;  // ��һ��ȫ��Ŀ��ID
    ConstantVelocityModel motion_model_;  // �˶�ģ��

    // ��������
    double position_weight_;          // λ��Ȩ��
    double velocity_weight_;          // �ٶ�Ȩ��
    double class_weight_;             // ���Ȩ��
    double feature_weight_;           // ����Ȩ��
    double max_position_distance_;    // ���λ�þ�����ֵ(��)
    double max_velocity_diff_;        // ����ٶȲ���ֵ(��/��)
    double min_similarity_threshold_; // ��С���ƶ���ֵ
};