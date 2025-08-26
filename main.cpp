#include "MultimodalDetectorSystem.h"
#include "Config.h"
#include <iostream>
#include <random>

// 生成随机特征向量
FeatureVector generateRandomFeatures(SensorType type) {
    FeatureVector feat;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    // 根据传感器类型生成不同的特征
    switch (type) {
    case SensorType::VISIBLE_CAMERA:
        // 视觉特征：20维
        for (int i = 0; i < 20; ++i) {
            feat.visual_features.push_back(std::abs(dist(gen)));
        }
        // 形状特征：10维
        for (int i = 0; i < 10; ++i) {
            feat.shape_features.push_back(std::abs(dist(gen)));
        }
        break;

    case SensorType::RADAR:
        // 雷达特征：15维
        for (int i = 0; i < 15; ++i) {
            feat.radar_features.push_back(std::abs(dist(gen)));
        }
        // 运动特征：5维
        for (int i = 0; i < 5; ++i) {
            feat.motion_features.push_back(std::abs(dist(gen)));
        }
        break;

    case SensorType::AUDIO:
        // 音频特征：25维
        for (int i = 0; i < 25; ++i) {
            feat.audio_features.push_back(std::abs(dist(gen)));
        }
        break;

    case SensorType::BEIDOU:
        // 运动特征：5维
        for (int i = 0; i < 5; ++i) {
            feat.motion_features.push_back(std::abs(dist(gen)));
        }
        break;

    case SensorType::INFRARED:
        // 视觉特征：15维
        for (int i = 0; i < 15; ++i) {
            feat.visual_features.push_back(std::abs(dist(gen)));
        }
        // 形状特征：8维
        for (int i = 0; i < 8; ++i) {
            feat.shape_features.push_back(std::abs(dist(gen)));
        }
        break;
    }

    return feat;
}

// 生成随机检测结果用于测试
std::vector<Detection> generateRandomDetections() {
    std::vector<Detection> detections;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> pos_dist(-100.0, 100.0);
    std::uniform_real_distribution<double> vel_dist(-10.0, 10.0);
    std::uniform_real_distribution<double> conf_dist(0.5, 1.0);
    std::uniform_int_distribution<int> class_dist(1, 5);  // 1-5对应PERSON到STATIC
    std::uniform_int_distribution<int> id_dist(1, 10);
    std::uniform_int_distribution<int> time_jitter(-50000, 50000);  // 时间抖动（微秒）

    // 传感器列表
    std::vector<std::pair<std::string, SensorType>> sensors = {
        {"camera_0", SensorType::VISIBLE_CAMERA},
        {"radar_0", SensorType::RADAR},
        {"audio_0", SensorType::AUDIO},
        {"beidou_0", SensorType::BEIDOU},
        {"infrared_0", SensorType::INFRARED}
    };

    // 为每个传感器生成随机检测结果
    for (const auto& [sensor_id, sensor_type] : sensors) {
        int num_detections = 1 + gen() % 3;  // 每个传感器1-3个检测结果

        int is_same = gen() % 2;

        if (is_same && detections.size() > 0) {
			std::cout << "生成与已有检测结果相似的检测，测试融合效果" << std::endl;
            Detection base = detections.back();
            for (int i = 0; i < num_detections; ++i) {
                Detection det = base;
                det.sensor_id = sensor_id;
                det.sensor_type = sensor_type;
                auto base_time = std::chrono::system_clock::now();
                det.timestamp = base_time + std::chrono::microseconds(time_jitter(gen));
                det.local_id = id_dist(gen);
                det.detected_class = static_cast<ObjectClass>(class_dist(gen));
                det.class_confidence = conf_dist(gen);
                det.detection_confidence = conf_dist(gen);
                // 生成特征向量
                det.features = generateRandomFeatures(sensor_type);
                detections.push_back(det);
            }
            continue;
        }

        for (int i = 0; i < num_detections; ++i) {
            Detection det;
            det.sensor_id = sensor_id;
            det.sensor_type = sensor_type;

            // 时间戳：当前时间 ± 50ms
            auto base_time = std::chrono::system_clock::now();
            det.timestamp = base_time + std::chrono::microseconds(time_jitter(gen));

            det.local_id = id_dist(gen);
            det.detected_class = static_cast<ObjectClass>(class_dist(gen));
            det.class_confidence = conf_dist(gen);

            // 位置和速度
            det.position = Eigen::Vector3d(pos_dist(gen), pos_dist(gen), pos_dist(gen));
            det.velocity = Eigen::Vector3d(vel_dist(gen), vel_dist(gen), vel_dist(gen));

            // 边界框（仅图像类传感器）
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

            // 生成特征向量
            det.features = generateRandomFeatures(sensor_type);

            detections.push_back(det);
        }
    }

    return detections;
}

int main() {
    Config::GetInstance().setPositionWeight(0.4);
    Config::GetInstance().setVelocityWeight(0.2);
    Config::GetInstance().setClassWeight(0.2);
    Config::GetInstance().setFeatureWeight(0.2);
    Config::GetInstance().setMaxPositionDistance(10.0);
    Config::GetInstance().setMaxVelocityDiff(5.0);
    Config::GetInstance().setMinSimilarityThreshold(0.5);
    Config::GetInstance().setConfThreshold(0.5);
    Config::GetInstance().setKlDivThreshold(0.1);
    Config::GetInstance().setNewTargetCosineThresh(0.6);
    Config::GetInstance().setNewSampleClusterThreshold(30);
    Config::GetInstance().setDpcRhoMin(5.0);
    Config::GetInstance().setDpcDeltaMin(0.3);
    Config::GetInstance().setCutDistance(0.25);
    Config::GetInstance().setIsolatedPointMin(0.75);
    Config::GetInstance().setIncrementalGamma(0.5);
    Config::GetInstance().setDistillLambda1(0.7);
    Config::GetInstance().setDistillLambda2(0.3);
	Config::GetInstance().setTemperature(5.0);
    Config::GetInstance().setHistoricalAccThreshold(0.9);
    Config::GetInstance().setNewClassAccThreshold(0.85);
    Config::GetInstance().setTimeGap(300);
	Config::GetInstance().setTeacherModelPath("teacher_model.pt");
	Config::GetInstance().setStudentModelPath("student_model.pt");

    // 1. 配置传感器校准参数
    std::map<std::string, SensorCalibration> calibrations;

    // 可见光相机校准
    SensorCalibration cam_calib;
    cam_calib.sensor_id = "camera_0";
    cam_calib.type = SensorType::VISIBLE_CAMERA;
    cam_calib.rotation = Eigen::Matrix3d::Identity();
    cam_calib.translation = Eigen::Vector3d(0.5, 0.0, 1.5);  // 相机位置偏移
    cam_calib.covariance = Eigen::Matrix3d::Identity() * 0.1;  // 测量噪声
    cam_calib.time_offset = std::chrono::microseconds(10000);  // 10ms偏移
    cam_calib.time_drift = 0.1;  // 0.1ppm漂移
    calibrations[cam_calib.sensor_id] = cam_calib;

    // 雷达校准
    SensorCalibration radar_calib;
    radar_calib.sensor_id = "radar_0";
    radar_calib.type = SensorType::RADAR;
    radar_calib.rotation = Eigen::Matrix3d::Identity();
    radar_calib.translation = Eigen::Vector3d(1.0, 0.0, 0.5);  // 雷达位置偏移
    radar_calib.covariance = Eigen::Matrix3d::Identity() * 0.5;  // 雷达噪声稍大
    radar_calib.time_offset = std::chrono::microseconds(-5000);  // -5ms偏移
    radar_calib.time_drift = 0.2;  // 0.2ppm漂移
    calibrations[radar_calib.sensor_id] = radar_calib;

    // 音频传感器校准
    SensorCalibration audio_calib;
    audio_calib.sensor_id = "audio_0";
    audio_calib.type = SensorType::AUDIO;
    audio_calib.rotation = Eigen::Matrix3d::Identity();
    audio_calib.translation = Eigen::Vector3d(0.0, 0.0, 1.2);  // 音频传感器位置
    audio_calib.covariance = Eigen::Matrix3d::Identity() * 1.0;  // 音频定位噪声较大
    audio_calib.time_offset = std::chrono::microseconds(5000);  // 5ms偏移
    audio_calib.time_drift = 0.5;  // 0.5ppm漂移
    calibrations[audio_calib.sensor_id] = audio_calib;

    // 北斗校准
    SensorCalibration beidou_calib;
    beidou_calib.sensor_id = "beidou_0";
    beidou_calib.type = SensorType::BEIDOU;
    beidou_calib.rotation = Eigen::Matrix3d::Identity();
    beidou_calib.translation = Eigen::Vector3d(0.0, 0.0, 0.0);  // 北斗作为参考
    beidou_calib.covariance = Eigen::Matrix3d::Identity() * 2.0;  // 北斗定位精度
    beidou_calib.time_offset = std::chrono::microseconds(0);  // 北斗时间较准
    beidou_calib.time_drift = 0.01;  // 北斗时间漂移小
    calibrations[beidou_calib.sensor_id] = beidou_calib;

    // 红外相机校准
    SensorCalibration ir_calib;
    ir_calib.sensor_id = "infrared_0";
    ir_calib.type = SensorType::INFRARED;
    ir_calib.rotation = Eigen::Matrix3d::Identity();
    ir_calib.translation = Eigen::Vector3d(0.5, 0.0, 1.6);  // 红外相机位置
    ir_calib.covariance = Eigen::Matrix3d::Identity() * 0.2;  // 红外噪声
    ir_calib.time_offset = std::chrono::microseconds(15000);  // 15ms偏移
    ir_calib.time_drift = 0.15;  // 0.15ppm漂移
    calibrations[ir_calib.sensor_id] = ir_calib;

    // 2. 创建多模态检测系统
    MultimodalDetectorSystem system(calibrations);

    // 3. 模拟处理多帧数据
    for (int frame = 0; frame < 5; ++frame) {
        std::cout << "\n处理第 " << frame + 1 << " 帧数据..." << std::endl;

        // 生成随机检测结果
        std::vector<Detection> detections = generateRandomDetections();
        std::cout << "生成了 " << detections.size() << " 个检测结果" << std::endl;

        // 添加检测结果到系统
        system.addDetections(detections);

        // 处理检测结果
        system.processDetections();

        // 获取融合结果
        const auto& fused_objects = system.getFusedObjects();
        std::cout << "融合得到 " << fused_objects.size() << " 个目标" << std::endl;

        // 保存结果到JSON文件
        std::string filename = "fusion_results_frame_" + std::to_string(frame) + ".json";
        if (system.saveResults(filename)) {
            std::cout << "结果已保存到 " << filename << std::endl;
        } else {
            std::cout << "保存结果失败" << std::endl;
        }

        // 等待一段时间模拟时间流逝
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "\n系统处理完成" << std::endl;
    return 0;
}