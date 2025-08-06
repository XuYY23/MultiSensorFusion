#pragma once

#include "enums.h"
#include "includes.h"

// 特征向量结构体，包含各种传感器的特征
struct FeatureVector {
    std::vector<double> visual_features;   // 视觉特征
    std::vector<double> radar_features;    // 雷达特征
    std::vector<double> audio_features;    // 音频特征
    std::vector<double> shape_features;    // 形状特征
    std::vector<double> motion_features;   // 运动特征

    // 检查特征向量是否为空
    bool isEmpty() const {
        return visual_features.empty() && radar_features.empty() &&
            audio_features.empty() && shape_features.empty() &&
            motion_features.empty();
    }
};

// 单个传感器的检测结果
struct Detection {
    std::string sensor_id;                  // 传感器ID
    SensorType sensor_type;                 // 传感器类型
    Timestamp timestamp;                    // 时间戳
    int local_id;                           // 传感器本地目标ID
    ObjectClass detected_class;             // 检测到的目标类别
    double class_confidence;                // 类别置信度
    Eigen::Vector3d position;               // 传感器坐标系下的位置
    Eigen::Vector3d position_global;        // 全局坐标系下的位置
    Eigen::Vector3d velocity;               // 传感器坐标系下的速度
    Eigen::Vector3d velocity_global;        // 全局坐标系下的速度
    cv::Rect2d bbox;                        // 边界框（图像类传感器）
    double detection_confidence;            // 检测置信度
    FeatureVector features;                 // 目标特征向量
    Eigen::Matrix3d covariance;             // 位置测量协方差矩阵
    Eigen::Matrix3d velocity_covariance;    // 速度测量协方差矩阵
};

// 融合后的目标
struct FusedObject {
    int global_id;                                      // 全局目标ID
    Timestamp timestamp;                                // 时间戳
    Eigen::Vector3d position;                           // 全局位置
    Eigen::Vector3d velocity;                           // 全局速度
    Eigen::Matrix3d position_covariance;                // 位置协方差
    Eigen::Matrix3d velocity_covariance;                // 速度协方差
    std::map<ObjectClass, double> class_probabilities;  // 类别概率分布
    FeatureVector fused_features;                       // 融合后的特征向量
    std::vector<Detection> associated_detections;       // 关联的检测结果
    int track_length;                                   // 跟踪长度（帧数）
    bool is_new;                                        // 是否为新目标
	ObjectClass final_class;                            // 最终类别
	double final_class_confidence;                      // 最终类别置信度
    bool has_category_conflict;                         // 是否存在类别冲突
    //double conflict_score;                              // 冲突评分（0~1，越高冲突越严重）
};

// 传感器校准参数
struct SensorCalibration {
	std::string sensor_id;                  // 传感器ID
	SensorType type;                        // 传感器类型
    Eigen::Matrix3d rotation;               // 旋转矩阵
    Eigen::Vector3d translation;            // 平移向量
    Eigen::Matrix3d covariance;             // covariance矩阵
    std::chrono::microseconds time_offset;  // 时间偏移
    double time_drift;                      // 时间漂移(ppm)，传感器的时间测量会随运行时间产生累积误差（漂移），通常用 ppm（百万分之一） 表示。例如，0.1ppm 表示每运行 1 秒，时间误差增加 0.1 微秒（1 秒 × 0.1/1e6）
};