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
    STATIC_OBSTACLE,
	POTENTIAL_NEW_TYPE  // Ǳ�������
};

class BaseObject {
public:
	virtual ~BaseObject() = default;

    // ��ObjectClassת��Ϊ�ַ���
    virtual std::string toString() {
        return "unknown";
    }

	// ���ڱȽ�����ObjectClass�Ƿ���ͬ
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