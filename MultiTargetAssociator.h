#pragma once

#include "ConstantVelocityModel.h"

// 多目标关联器，负责将检测结果与已有目标关联
class MultiTargetAssociator {
public:
    MultiTargetAssociator();
    ~MultiTargetAssociator();

    // 关联新检测结果与已有目标
    std::vector<FusedObject> associateTargets(const std::vector<Detection>& new_detections,
                                              const std::vector<FusedObject>& existing_targets,
                                              Timestamp current_time);

    std::vector<FusedObject> associateTargets(const std::vector<Detection>& detections, Timestamp current_time);

    // 计算两个检测结果的相似度
    double calculateSimilarity(const Detection& a, const Detection& b);

    // 计算特征向量的相似度
    double calculateFeatureSimilarity(const FeatureVector& a, const FeatureVector& b);

    // 计算两个向量的余弦相似度
    double cosineSimilarity(const Eigen::Vector3d& a, const Eigen::Vector3d& b);
    double cosineSimilarity(const std::vector<double>& a, const std::vector<double>& b);

	// 计算高斯相似度（基于位置和速度的马氏距离）
    double gaussianSimilarity(const double& dis, const double& scale);

private:
    // 使用dlib库实现的匈牙利算法
    std::vector<std::pair<int, int>> hungarianAlgorithm(const std::vector<std::vector<double>>& cost_matrix);

    int next_global_id_;  // 下一个全局目标ID
    ConstantVelocityModel motion_model_;  // 运动模型
};