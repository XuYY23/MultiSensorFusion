#pragma once

#include "enums.h"
#include "includes.h"

// ���������ṹ�壬�������ִ�����������
struct FeatureVector {
    std::vector<double> visual_features;   // �Ӿ�����
    std::vector<double> radar_features;    // �״�����
    std::vector<double> audio_features;    // ��Ƶ����
    std::vector<double> shape_features;    // ��״����
    std::vector<double> motion_features;   // �˶�����

    // ������������Ƿ�Ϊ��
    bool isEmpty() const {
        return visual_features.empty() && radar_features.empty() &&
            audio_features.empty() && shape_features.empty() &&
            motion_features.empty();
    }
};

// �����������ļ����
struct Detection {
    std::string sensor_id;                  // ������ID
    SensorType sensor_type;                 // ����������
    Timestamp timestamp;                    // ʱ���
    int local_id;                           // ����������Ŀ��ID
    ObjectClass detected_class;             // ��⵽��Ŀ�����
    double class_confidence;                // ������Ŷ�
    Eigen::Vector3d position;               // ����������ϵ�µ�λ��
    Eigen::Vector3d position_global;        // ȫ������ϵ�µ�λ��
    Eigen::Vector3d velocity;               // ����������ϵ�µ��ٶ�
    Eigen::Vector3d velocity_global;        // ȫ������ϵ�µ��ٶ�
    cv::Rect2d bbox;                        // �߽��ͼ���ഫ������
    double detection_confidence;            // ������Ŷ�
    FeatureVector features;                 // Ŀ����������
    Eigen::Matrix3d covariance;             // λ�ò���Э�������
    Eigen::Matrix3d velocity_covariance;    // �ٶȲ���Э�������
};

// �ںϺ��Ŀ��
struct FusedObject {
    int global_id;                                      // ȫ��Ŀ��ID
    Timestamp timestamp;                                // ʱ���
    Eigen::Vector3d position;                           // ȫ��λ��
    Eigen::Vector3d velocity;                           // ȫ���ٶ�
    Eigen::Matrix3d position_covariance;                // λ��Э����
    Eigen::Matrix3d velocity_covariance;                // �ٶ�Э����
    std::map<ObjectClass, double> class_probabilities;  // �����ʷֲ�
    FeatureVector fused_features;                       // �ںϺ����������
    std::vector<Detection> associated_detections;       // �����ļ����
    int track_length;                                   // ���ٳ��ȣ�֡����
    bool is_new;                                        // �Ƿ�Ϊ��Ŀ��
	ObjectClass final_class;                            // �������
	double final_class_confidence;                      // ����������Ŷ�
    bool has_category_conflict;                         // �Ƿ��������ͻ
    //double conflict_score;                              // ��ͻ���֣�0~1��Խ�߳�ͻԽ���أ�
};

// ������У׼����
struct SensorCalibration {
	std::string sensor_id;                  // ������ID
	SensorType type;                        // ����������
    Eigen::Matrix3d rotation;               // ��ת����
    Eigen::Vector3d translation;            // ƽ������
    Eigen::Matrix3d covariance;             // covariance����
    std::chrono::microseconds time_offset;  // ʱ��ƫ��
    double time_drift;                      // ʱ��Ư��(ppm)����������ʱ�������������ʱ������ۻ���Ư�ƣ���ͨ���� ppm�������֮һ�� ��ʾ�����磬0.1ppm ��ʾÿ���� 1 �룬ʱ��������� 0.1 ΢�루1 �� �� 0.1/1e6��
};