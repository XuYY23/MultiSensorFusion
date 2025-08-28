#pragma once

#include "structs.h"

// ʱ�ն����������𽫲�ͬ������������ͬ����ͳһʱ�������ϵ
class SpatioTemporalAligner {
public:
    SpatioTemporalAligner(const std::map<std::string, SensorCalibration>& calibrations);

    // �Ե�����������пռ���루����������->ȫ�����꣩
    void alignSpatial(Detection& detection);

    // �Ե������������ʱ����루������ʱ��->ϵͳʱ�䣩
    void alignTemporal(Detection& detection);

    // ��һ����������ʱ��ͬ���������ǲ�ֵ��Ŀ��ʱ���
    std::vector<Detection> synchronizeDetections(const std::vector<Detection>& detections, Timestamp target_time);

    // ��������ʱ���֮��ļ������ֵ
    Detection interpolateDetection(const Detection& earlier,
                                   const Detection& later,
                                   Timestamp target_time);

private:
    std::map<std::string, SensorCalibration> calibrations_;  // ������У׼����

    // ���������������������������Ĳ�ֵ
    FeatureVector interpolateFeatures(const FeatureVector& earlier,
                                      const FeatureVector& later,
                                      double alpha);

    // �����������������������Ĳ�ֵ
    std::vector<double> interpolateVector(const std::vector<double>& earlier,
                                          const std::vector<double>& later,
                                          double alpha);
};