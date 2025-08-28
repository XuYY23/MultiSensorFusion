#pragma once

#include "structs.h"

// JSON������������ںϽ��ת��ΪJSON��ʽ
class JsonOutput {
public:
    // �������ת��ΪJSON
    json detectionToJson(const Detection& detection);

    // ���ں�Ŀ��ת��ΪJSON
    json fusedObjectToJson(const FusedObject& object);

    // ������ں�Ŀ�걣��ΪJSON�ļ�
    bool saveResults(const std::vector<FusedObject>& objects, const std::string& filename);

private:
    // ��ʱ���ת��Ϊ�ַ���
    std::string timestampToString(Timestamp timestamp);

    // ��Eigen����ת��ΪJSON����
    json eigenToJson(const Eigen::Vector3d& vec);

    // ��Eigen����ת��ΪJSON����
    json eigenToJson(const Eigen::Matrix3d& mat);

    // ����������ת��ΪJSON
    json featureToJson(const FeatureVector& features);
};