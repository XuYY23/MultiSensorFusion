#pragma once

#include "structs.h"

// JSON输出器，负责将融合结果转换为JSON格式
class JsonOutput {
public:
    // 将检测结果转换为JSON
    json detectionToJson(const Detection& detection);

    // 将融合目标转换为JSON
    json fusedObjectToJson(const FusedObject& object);

    // 将多个融合目标保存为JSON文件
    bool saveResults(const std::vector<FusedObject>& objects, const std::string& filename);

private:
    // 将时间戳转换为字符串
    std::string timestampToString(Timestamp timestamp);

    // 将Eigen向量转换为JSON数组
    json eigenToJson(const Eigen::Vector3d& vec);

    // 将Eigen矩阵转换为JSON数组
    json eigenToJson(const Eigen::Matrix3d& mat);

    // 将特征向量转换为JSON
    json featureToJson(const FeatureVector& features);
};