#pragma once

#include "structs.h"

// 数据融合器，负责特征级和决策级融合
class DataFusion {
public:
    // 特征级融合：融合多个检测结果的特征
    FeatureVector fuseFeatures(const std::vector<Detection>& detections);

    // 决策级融合：融合多个检测结果的类别判断（基于D-S证据理论）
    std::map<ObjectClass, double> fuseDecisions(const std::vector<Detection>& detections, bool& has_category_conflict);

    // 位置融合：融合多个检测结果的位置信息
    Eigen::Vector3d fusePositions(const std::vector<Detection>& detections, Eigen::Matrix3d& covariance);

    // 速度融合：融合多个检测结果的速度信息
    Eigen::Vector3d fuseVelocities(const std::vector<Detection>& detections, Eigen::Matrix3d& covariance);
};
