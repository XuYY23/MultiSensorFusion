#pragma once

#include "structs.h"

// 时空对齐器，负责将不同传感器的数据同步到统一时间和坐标系
class SpatioTemporalAligner {
public:
    SpatioTemporalAligner(const std::map<std::string, SensorCalibration>& calibrations);

    // 对单个检测结果进行空间对齐（传感器坐标->全局坐标）
    void alignSpatial(Detection& detection);

    // 对单个检测结果进行时间对齐（传感器时间->系统时间）
    void alignTemporal(Detection& detection);

    // 对一组检测结果进行时间同步，将它们插值到目标时间点
    std::vector<Detection> synchronizeDetections(const std::vector<Detection>& detections, Timestamp target_time);

    // 计算两个时间点之间的检测结果插值
    Detection interpolateDetection(const Detection& earlier,
                                   const Detection& later,
                                   Timestamp target_time);

private:
    std::map<std::string, SensorCalibration> calibrations_;  // 传感器校准参数

    // 辅助函数：计算两个特征向量的插值
    FeatureVector interpolateFeatures(const FeatureVector& earlier,
                                      const FeatureVector& later,
                                      double alpha);

    // 辅助函数：计算两个向量的插值
    std::vector<double> interpolateVector(const std::vector<double>& earlier,
                                          const std::vector<double>& later,
                                          double alpha);
};