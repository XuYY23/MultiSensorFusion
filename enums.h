#pragma once

#include <string>

// ����������ö��
enum class SensorType {
    VISIBLE_CAMERA,   // �ɼ������
    RADAR,            // �״�
    AUDIO,            // ����������
    BEIDOU,           // ������λ
    INFRARED          // �������
};

// Ŀ�������
enum class ObjectClass {
    UNKNOWN,
    PERSON,
    VEHICLE,
    BICYCLE,
    ANIMAL,
    STATIC_OBSTACLE
};

// ��ObjectClassת��Ϊ�ַ���
std::string objectClassToString(ObjectClass cls);

// ���ַ���ת��ΪObjectClass
ObjectClass stringToObjectClass(const std::string& str);