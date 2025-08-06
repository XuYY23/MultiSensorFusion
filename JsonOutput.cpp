#include "JsonOutput.h"

// ��ʱ���ת��Ϊ�ַ���
std::string JsonOutput::timestampToString(Timestamp timestamp) {
    auto time = std::chrono::system_clock::to_time_t(timestamp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        timestamp.time_since_epoch() % std::chrono::seconds(1)
    );

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S")
        << "." << std::setw(3) << std::setfill('0') << ms.count();
    return ss.str();
}

// ��Eigen����ת��ΪJSON����
json JsonOutput::eigenToJson(const Eigen::Vector3d& vec) {
    return { vec.x(), vec.y(), vec.z() };
}

// ��Eigen����ת��ΪJSON����
json JsonOutput::eigenToJson(const Eigen::Matrix3d& mat) {
    return {
        {mat(0,0), mat(0,1), mat(0,2)},
        {mat(1,0), mat(1,1), mat(1,2)},
        {mat(2,0), mat(2,1), mat(2,2)}
    };
}

// ����������ת��ΪJSON
json JsonOutput::featureToJson(const FeatureVector& features) {
    json j;

    if (!features.visual_features.empty()) {
        j["visual"] = features.visual_features;
    }
    if (!features.radar_features.empty()) {
        j["radar"] = features.radar_features;
    }
    if (!features.audio_features.empty()) {
        j["audio"] = features.audio_features;
    }
    if (!features.shape_features.empty()) {
        j["shape"] = features.shape_features;
    }
    if (!features.motion_features.empty()) {
        j["motion"] = features.motion_features;
    }

    return j;
}

// �������ת��ΪJSON
json JsonOutput::detectionToJson(const Detection& detection) {
    json j;

    j["sensor_id"] = detection.sensor_id;
    j["sensor_type"] = static_cast<int>(detection.sensor_type);
    j["timestamp"] = timestampToString(detection.timestamp);
    j["local_id"] = detection.local_id;
    j["class"] = static_cast<int>(detection.detected_class);
    j["class_name"] = [&]() {
        switch (detection.detected_class) {
            case ObjectClass::PERSON: 
                return "PERSON";
            case ObjectClass::VEHICLE: 
                return "VEHICLE";
            case ObjectClass::BICYCLE: 
                return "BICYCLE";
            case ObjectClass::ANIMAL: 
                return "ANIMAL";
            case ObjectClass::STATIC_OBSTACLE: 
                return "STATIC_OBSTACLE";
            default: 
                return "UNKNOWN";
        }
    }();
    j["class_confidence"] = detection.class_confidence;
    j["position"] = eigenToJson(detection.position);
    j["position_global"] = eigenToJson(detection.position_global);
    j["velocity"] = eigenToJson(detection.velocity);
    j["velocity_global"] = eigenToJson(detection.velocity_global);
    j["detection_confidence"] = detection.detection_confidence;

    if (detection.bbox.area() > 0) {
        j["bbox"] = {
            detection.bbox.x,
            detection.bbox.y,
            detection.bbox.width,
            detection.bbox.height
        };
    }

    j["features"] = featureToJson(detection.features);

    return j;
}

// ���ں�Ŀ��ת��ΪJSON
json JsonOutput::fusedObjectToJson(const FusedObject& object) {
    json j;

    j["global_id"] = object.global_id;
    j["timestamp"] = timestampToString(object.timestamp);
    j["position"] = eigenToJson(object.position);
    j["velocity"] = eigenToJson(object.velocity);
    j["position_covariance"] = eigenToJson(object.position_covariance);
    j["velocity_covariance"] = eigenToJson(object.velocity_covariance);
    j["track_length"] = object.track_length;
    j["is_new"] = object.is_new;

    // ������
    for (const auto& [cls, prob] : object.class_probabilities) {
        std::string cls_name;
        switch (cls) {
            case ObjectClass::PERSON: 
                cls_name = "PERSON"; 
                break;
            case ObjectClass::VEHICLE: 
                cls_name = "VEHICLE";
                break;
            case ObjectClass::BICYCLE: 
                cls_name = "BICYCLE";
                break;
            case ObjectClass::ANIMAL: 
                cls_name = "ANIMAL"; 
                break;
            case ObjectClass::STATIC_OBSTACLE: 
                cls_name = "STATIC_OBSTACLE"; 
                break;
            default: 
                cls_name = "UNKNOWN";
                break;
        }
        j["class_probabilities"][cls_name] = prob;
    }

    // �ں�����
    j["fused_features"] = featureToJson(object.fused_features);

    // �����ļ����
    j["associated_detections"] = json::array();
    for (const auto& det : object.associated_detections) {
        j["associated_detections"].push_back(detectionToJson(det));
    }

    return j;
}

// ������ں�Ŀ�걣��ΪJSON�ļ�
bool JsonOutput::saveResults(const std::vector<FusedObject>& objects, const std::string& filename) {
    json j;

    j["timestamp"] = timestampToString(std::chrono::system_clock::now());
    j["num_objects"] = objects.size();
    j["objects"] = json::array();

    for (const auto& obj : objects) {
        j["objects"].push_back(fusedObjectToJson(obj));
    }

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        return false;
    }

    ofs << std::setw(4) << j << std::endl;
    ofs.close();

    return true;
}