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
    STATIC_OBSTACLE,
	POTENTIAL_NEW_TYPE  // 潜在新类别
};

class BaseObject {
public:
	virtual ~BaseObject() = default;

    // 将ObjectClass转换为字符串
    virtual std::string toString() {
        return "unknown";
    }

	// 用于比较两个ObjectClass是否相同
	virtual bool operator==(const ObjectClass& other) const {
        return other == ObjectClass::UNKNOWN;
    }

    virtual ObjectClass toEnum() const {
        return ObjectClass::UNKNOWN;
	}
};

class PersonObject : public BaseObject {
public:
    std::string toString() override {
        return "person";
	}

    bool operator==(const ObjectClass& other) const override {
        return other == ObjectClass::PERSON;
	}

    ObjectClass toEnum() const override {
        return ObjectClass::PERSON;
    }
};

class VehicleObject : public BaseObject {
public:
    std::string toString() override {
        return "vehicle";
    }

    bool operator==(const ObjectClass& other) const override {
        return other == ObjectClass::VEHICLE;
    }

    ObjectClass toEnum() const override {
        return ObjectClass::VEHICLE;
	}
};

class BicycleObject : public BaseObject {
public:
    std::string toString() override {
        return "bicycle";
    }

    bool operator==(const ObjectClass& other) const override {
        return other == ObjectClass::BICYCLE;
    }

    ObjectClass toEnum() const override {
        return ObjectClass::BICYCLE;
    }
};

class AnimalObject : public BaseObject {
public:
    std::string toString() override {
        return "animal";
    }

    bool operator==(const ObjectClass& other) const override {
        return other == ObjectClass::ANIMAL;
	}

    ObjectClass toEnum() const override {
        return ObjectClass::ANIMAL;
	}
};

class StaticObstacleObject : public BaseObject {
public:
    std::string toString() override {
        return "static_obstacle";
    }

    bool operator==(const ObjectClass& other) const override {
        return other == ObjectClass::STATIC_OBSTACLE;
	}

    ObjectClass toEnum() const override {
        return ObjectClass::STATIC_OBSTACLE;
	}
};

class PotentialNewTypeObject : public BaseObject {
public:
    std::string toString() override {
        return "potential_new_type";
    }
    bool operator==(const ObjectClass& other) const override {
        return other == ObjectClass::POTENTIAL_NEW_TYPE;
	}

    ObjectClass toEnum() const override {
        return ObjectClass::POTENTIAL_NEW_TYPE;
	}
};