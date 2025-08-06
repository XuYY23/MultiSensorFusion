#pragma once

#include <string>

// 传感器类型枚举
enum class SensorType {
    VISIBLE_CAMERA,   // 可见光相机
    RADAR,            // 雷达
    AUDIO,            // 声音传感器
    BEIDOU,           // 北斗定位
    INFRARED          // 红外成像
};

// 目标类别定义
enum class ObjectClass {
    UNKNOWN,
    PERSON,
    VEHICLE,
    BICYCLE,
    ANIMAL,
    STATIC_OBSTACLE
};

// 将ObjectClass转换为字符串
std::string objectClassToString(ObjectClass cls);

// 从字符串转换为ObjectClass
ObjectClass stringToObjectClass(const std::string& str);