#include "MultimodalDetectorSystem.h"
#include <iostream>
#include <random>

// ���������������
FeatureVector generateRandomFeatures(SensorType type) {
    FeatureVector feat;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    // ���ݴ������������ɲ�ͬ������
    switch (type) {
    case SensorType::VISIBLE_CAMERA:
        // �Ӿ�������20ά
        for (int i = 0; i < 20; ++i) {
            feat.visual_features.push_back(std::abs(dist(gen)));
        }
        // ��״������10ά
        for (int i = 0; i < 10; ++i) {
            feat.shape_features.push_back(std::abs(dist(gen)));
        }
        break;

    case SensorType::RADAR:
        // �״�������15ά
        for (int i = 0; i < 15; ++i) {
            feat.radar_features.push_back(std::abs(dist(gen)));
        }
        // �˶�������5ά
        for (int i = 0; i < 5; ++i) {
            feat.motion_features.push_back(std::abs(dist(gen)));
        }
        break;

    case SensorType::AUDIO:
        // ��Ƶ������25ά
        for (int i = 0; i < 25; ++i) {
            feat.audio_features.push_back(std::abs(dist(gen)));
        }
        break;

    case SensorType::BEIDOU:
        // �˶�������5ά
        for (int i = 0; i < 5; ++i) {
            feat.motion_features.push_back(std::abs(dist(gen)));
        }
        break;

    case SensorType::INFRARED:
        // �Ӿ�������15ά
        for (int i = 0; i < 15; ++i) {
            feat.visual_features.push_back(std::abs(dist(gen)));
        }
        // ��״������8ά
        for (int i = 0; i < 8; ++i) {
            feat.shape_features.push_back(std::abs(dist(gen)));
        }
        break;
    }

    return feat;
}

// ���������������ڲ���
std::vector<Detection> generateRandomDetections() {
    std::vector<Detection> detections;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> pos_dist(-100.0, 100.0);
    std::uniform_real_distribution<double> vel_dist(-10.0, 10.0);
    std::uniform_real_distribution<double> conf_dist(0.5, 1.0);
    std::uniform_int_distribution<int> class_dist(1, 5);  // 1-5��ӦPERSON��STATIC
    std::uniform_int_distribution<int> id_dist(1, 10);
    std::uniform_int_distribution<int> time_jitter(-50000, 50000);  // ʱ�䶶����΢�룩

    // �������б�
    std::vector<std::pair<std::string, SensorType>> sensors = {
        {"camera_0", SensorType::VISIBLE_CAMERA},
        {"radar_0", SensorType::RADAR},
        {"audio_0", SensorType::AUDIO},
        {"beidou_0", SensorType::BEIDOU},
        {"infrared_0", SensorType::INFRARED}
    };

    // Ϊÿ��������������������
    for (const auto& [sensor_id, sensor_type] : sensors) {
        int num_detections = 1 + gen() % 3;  // ÿ��������1-3�������

        for (int i = 0; i < num_detections; ++i) {
            Detection det;
            det.sensor_id = sensor_id;
            det.sensor_type = sensor_type;

            // ʱ�������ǰʱ�� �� 50ms
            auto base_time = std::chrono::system_clock::now();
            det.timestamp = base_time + std::chrono::microseconds(time_jitter(gen));

            det.local_id = id_dist(gen);
            det.detected_class = static_cast<ObjectClass>(class_dist(gen));
            det.class_confidence = conf_dist(gen);

            // λ�ú��ٶ�
            det.position = Eigen::Vector3d(pos_dist(gen), pos_dist(gen), pos_dist(gen));
            det.velocity = Eigen::Vector3d(vel_dist(gen), vel_dist(gen), vel_dist(gen));

            // �߽�򣨽�ͼ���ഫ������
            if (sensor_type == SensorType::VISIBLE_CAMERA ||
                sensor_type == SensorType::INFRARED) {
                det.bbox = cv::Rect2d(
                    50 + gen() % 500,
                    50 + gen() % 300,
                    30 + gen() % 100,
                    50 + gen() % 150
                );
            }

            det.detection_confidence = conf_dist(gen);

            // ������������
            det.features = generateRandomFeatures(sensor_type);

            detections.push_back(det);
        }
    }

    return detections;
}

int main() {
    std::cout << "��ģ̬�ഫ����Ŀ�����ں�ϵͳʾ��" << std::endl;

    // 1. ���ô�����У׼����
    std::map<std::string, SensorCalibration> calibrations;

    // �ɼ������У׼
    SensorCalibration cam_calib;
    cam_calib.sensor_id = "camera_0";
    cam_calib.type = SensorType::VISIBLE_CAMERA;
    cam_calib.rotation = Eigen::Matrix3d::Identity();
    cam_calib.translation = Eigen::Vector3d(0.5, 0.0, 1.5);  // ���λ��ƫ��
    cam_calib.covariance = Eigen::Matrix3d::Identity() * 0.1;  // ��������
    cam_calib.time_offset = std::chrono::microseconds(10000);  // 10msƫ��
    cam_calib.time_drift = 0.1;  // 0.1ppmƯ��
    calibrations[cam_calib.sensor_id] = cam_calib;

    // �״�У׼
    SensorCalibration radar_calib;
    radar_calib.sensor_id = "radar_0";
    radar_calib.type = SensorType::RADAR;
    radar_calib.rotation = Eigen::Matrix3d::Identity();
    radar_calib.translation = Eigen::Vector3d(1.0, 0.0, 0.5);  // �״�λ��ƫ��
    radar_calib.covariance = Eigen::Matrix3d::Identity() * 0.5;  // �״������Դ�
    radar_calib.time_offset = std::chrono::microseconds(-5000);  // -5msƫ��
    radar_calib.time_drift = 0.2;  // 0.2ppmƯ��
    calibrations[radar_calib.sensor_id] = radar_calib;

    // ��Ƶ������У׼
    SensorCalibration audio_calib;
    audio_calib.sensor_id = "audio_0";
    audio_calib.type = SensorType::AUDIO;
    audio_calib.rotation = Eigen::Matrix3d::Identity();
    audio_calib.translation = Eigen::Vector3d(0.0, 0.0, 1.2);  // ��Ƶ������λ��
    audio_calib.covariance = Eigen::Matrix3d::Identity() * 1.0;  // ��Ƶ��λ�����ϴ�
    audio_calib.time_offset = std::chrono::microseconds(5000);  // 5msƫ��
    audio_calib.time_drift = 0.5;  // 0.5ppmƯ��
    calibrations[audio_calib.sensor_id] = audio_calib;

    // ����У׼
    SensorCalibration beidou_calib;
    beidou_calib.sensor_id = "beidou_0";
    beidou_calib.type = SensorType::BEIDOU;
    beidou_calib.rotation = Eigen::Matrix3d::Identity();
    beidou_calib.translation = Eigen::Vector3d(0.0, 0.0, 0.0);  // ������Ϊ�ο�
    beidou_calib.covariance = Eigen::Matrix3d::Identity() * 2.0;  // ������λ����
    beidou_calib.time_offset = std::chrono::microseconds(0);  // ����ʱ���׼
    beidou_calib.time_drift = 0.01;  // ����ʱ��Ư��С
    calibrations[beidou_calib.sensor_id] = beidou_calib;

    // �������У׼
    SensorCalibration ir_calib;
    ir_calib.sensor_id = "infrared_0";
    ir_calib.type = SensorType::INFRARED;
    ir_calib.rotation = Eigen::Matrix3d::Identity();
    ir_calib.translation = Eigen::Vector3d(0.5, 0.0, 1.6);  // �������λ��
    ir_calib.covariance = Eigen::Matrix3d::Identity() * 0.2;  // ��������
    ir_calib.time_offset = std::chrono::microseconds(15000);  // 15msƫ��
    ir_calib.time_drift = 0.15;  // 0.15ppmƯ��
    calibrations[ir_calib.sensor_id] = ir_calib;

    // 2. ������ģ̬���ϵͳ
    MultimodalDetectorSystem system(calibrations);

    // 3. ģ�⴦���֡����
    for (int frame = 0; frame < 5; ++frame) {
        std::cout << "\n����� " << frame + 1 << " ֡����..." << std::endl;

        // ������������
        std::vector<Detection> detections = generateRandomDetections();
        std::cout << "������ " << detections.size() << " �������" << std::endl;

        // ��Ӽ������ϵͳ
        system.addDetections(detections);

        // ��������
        system.processDetections();

        // ��ȡ�ںϽ��
        const auto& fused_objects = system.getFusedObjects();
        std::cout << "�ںϵõ� " << fused_objects.size() << " ��Ŀ��" << std::endl;

        // ��������JSON�ļ�
        std::string filename = "fusion_results_frame_" + std::to_string(frame) + ".json";
        if (system.saveResults(filename)) {
            std::cout << "����ѱ��浽 " << filename << std::endl;
        }
        else {
            std::cout << "������ʧ��" << std::endl;
        }

        // �ȴ�һ��ʱ��ģ��ʱ������
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "\nϵͳ�������" << std::endl;
    return 0;
}