#pragma once

#include "SpatioTemporalAligner.h"
#include "FeatureExtractor.h"
#include "MultiTargetAssociator.h"
#include "DataFusion.h"
#include "JsonOutput.h"

// ��ģ̬�ഫ����Ŀ�����ں�ϵͳ����
class MultimodalDetectorSystem {
public:
    MultimodalDetectorSystem(const std::map<std::string, SensorCalibration>& calibrations);

    // ����µļ����
    void addDetections(const std::vector<Detection>& detections);

    // ����ǰ���м����
    void processDetections();

    // ��ȡ��ǰ�ںϽ��
    const std::vector<FusedObject>& getFusedObjects() const;

    // �����ںϽ����JSON�ļ�
    bool saveResults(const std::string& filename);

    // ����ʱ��ͬ����Ŀ��ʱ�䣨�����������ʹ�����¼��ʱ�䣩
    void setTargetTimestamp(Timestamp target_time);

private:
    SpatioTemporalAligner aligner_;       // ʱ�ն�����
    MultiTargetAssociator associator_;    // ��Ŀ�������
    DataFusion fusion_;                   // �����ں���
    JsonOutput json_output_;              // JSON�����

    std::vector<Detection> current_detections_;  // ��ǰ���������
    std::vector<FusedObject> fused_objects_;     // �ںϺ��Ŀ��

    Timestamp target_timestamp_;  // ʱ��ͬ����Ŀ��ʱ��
    bool use_custom_target_time_; // �Ƿ�ʹ���Զ���Ŀ��ʱ��
};