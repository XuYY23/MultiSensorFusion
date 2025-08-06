#include "enums.h"

// 将ObjectClass转换为字符串
std::string objectClassToString(ObjectClass cls) {
    switch (cls) {
        case ObjectClass::PERSON: 
            return "person";
        case ObjectClass::VEHICLE: 
            return "vehicle";
        case ObjectClass::BICYCLE: 
            return "bicycle";
        case ObjectClass::ANIMAL: 
            return "animal";
        case ObjectClass::STATIC_OBSTACLE: 
            return "static_obstacle";
        default:
            return "unknown";
    }
}

// 从字符串转换为ObjectClass
ObjectClass stringToObjectClass(const std::string& str) {
    if (str == "person") {
        return ObjectClass::PERSON;
    }

    if (str == "vehicle") {
        return ObjectClass::VEHICLE;
    }

    if (str == "bicycle") {
        return ObjectClass::BICYCLE;
    }

    if (str == "animal") {
        return ObjectClass::ANIMAL;
    }

    if (str == "static_obstacle") {
        return ObjectClass::STATIC_OBSTACLE;
    }

    return ObjectClass::UNKNOWN;
}