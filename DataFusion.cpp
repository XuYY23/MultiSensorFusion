#include "DataFusion.h"

// 特征级融合：融合多个检测结果的特征
FeatureVector DataFusion::fuseFeatures(const std::vector<Detection>& detections) {
    FeatureVector fused;
    if (detections.empty()) {
        return fused;
    }

    // 对每种特征进行平均融合
    std::vector<std::vector<double>> all_visual, all_radar, all_audio, all_shape, all_motion;

    // 收集所有特征
    for (const auto& det : detections) {
        if (!det.features.visual_features.empty()) {
            all_visual.push_back(det.features.visual_features);
        }
        if (!det.features.radar_features.empty()) {
            all_radar.push_back(det.features.radar_features);
        }
        if (!det.features.audio_features.empty()) {
            all_audio.push_back(det.features.audio_features);
        }
        if (!det.features.shape_features.empty()) {
            all_shape.push_back(det.features.shape_features);
        }
        if (!det.features.motion_features.empty()) {
            all_motion.push_back(det.features.motion_features);
        }
    }

    // 计算视觉特征平均值
    if (!all_visual.empty()) {
        size_t dim = all_visual[0].size();
        fused.visual_features.resize(dim, 0.0);
        for (const auto& feat : all_visual) {
            for (size_t i = 0; i < dim && i < feat.size(); ++i) {
                fused.visual_features[i] += feat[i] / all_visual.size();
            }
        }
    }

    // 计算雷达特征平均值
    if (!all_radar.empty()) {
        size_t dim = all_radar[0].size();
        fused.radar_features.resize(dim, 0.0);
        for (const auto& feat : all_radar) {
            for (size_t i = 0; i < dim && i < feat.size(); ++i) {
                fused.radar_features[i] += feat[i] / all_radar.size();
            }
        }
    }

    // 计算音频特征平均值
    if (!all_audio.empty()) {
        size_t dim = all_audio[0].size();
        fused.audio_features.resize(dim, 0.0);
        for (const auto& feat : all_audio) {
            for (size_t i = 0; i < dim && i < feat.size(); ++i) {
                fused.audio_features[i] += feat[i] / all_audio.size();
            }
        }
    }

    // 计算形状特征平均值
    if (!all_shape.empty()) {
        size_t dim = all_shape[0].size();
        fused.shape_features.resize(dim, 0.0);
        for (const auto& feat : all_shape) {
            for (size_t i = 0; i < dim && i < feat.size(); ++i) {
                fused.shape_features[i] += feat[i] / all_shape.size();
            }
        }
    }

    // 计算运动特征平均值
    if (!all_motion.empty()) {
        size_t dim = all_motion[0].size();
        fused.motion_features.resize(dim, 0.0);
        for (const auto& feat : all_motion) {
            for (size_t i = 0; i < dim && i < feat.size(); ++i) {
                fused.motion_features[i] += feat[i] / all_motion.size();
            }
        }
    }

    return fused;
}

// 决策级融合：融合多个检测结果的类别判断（基于D-S证据理论）
std::map<ObjectClass, double> DataFusion::fuseDecisions(const std::vector<Detection>& detections, double conf_threshold, bool& has_category_conflict) {
    std::map<ObjectClass, double> fused;
    if (detections.empty()) {
        return fused;
    }

    // 初始化基本概率分配
    std::vector<std::map<ObjectClass, double>> bpas;
	std::set<ObjectClass> clases;
    for (const auto& det : detections) {
        if (det.detection_confidence < conf_threshold) {
            continue;  // 如果置信度低于阈值，跳过该检测
        }
        std::map<ObjectClass, double> bpa;
        // 对于检测到的类别，分配与置信度相关的概率
        double confidence = det.class_confidence;
        bpa[det.detected_class] = confidence * 0.9;  // 90%分配给检测类别
        bpa[ObjectClass::UNKNOWN] = 1.0 - confidence * 0.9;  // 剩余部分为不确定
        if(det.detected_class != ObjectClass::UNKNOWN) {
            clases.insert(det.detected_class);
		}
        bpas.push_back(bpa);
    }

    if(clases.size() > 1 || clases.size() == 0) {
        has_category_conflict = true;  // 存在多个类别或没有可信的检测结果，标记为冲突
		return std::map<ObjectClass, double>();
    } else {
        has_category_conflict = false; // 只有一个类别，没有冲突
	}

    // 如果只有一个检测结果，直接返回其BPA
    if (bpas.size() == 1) {
        return bpas[0];
    }

    // Dempster组合规则融合第一个和第二个BPA
    std::map<ObjectClass, double> combined = bpas[0];
    for (size_t i = 1; i < bpas.size(); ++i) {
        std::map<ObjectClass, double> temp;
        double conflict = 0.0;
        
        // 计算所有可能的交集
        for (const auto& [a, mass_a] : combined) {
            for (const auto& [b, mass_b] : bpas[i]) {
                if (a == b) {
                    // 同一类别，直接相乘
                    temp[a] += mass_a * mass_b;
                } else if (a != ObjectClass::UNKNOWN && b != ObjectClass::UNKNOWN) {
                    // 不同类别，属于冲突
                    conflict += mass_a * mass_b;
                } else {
                    // 一个是UNKNOWN，结果为非UNKNOWN的类别
                    ObjectClass non_unknown = (a != ObjectClass::UNKNOWN) ? a : b;
                    temp[non_unknown] += mass_a * mass_b;
                }
            }
        }

        // 归一化（除以1 - conflict）
        if (conflict < 1.0) {
            double scale = 1.0 / (1.0 - conflict);
            for (auto& [cls, mass] : temp) {
                mass *= scale;
            }
        }

        combined = temp;
    }

    return combined;
}

// 位置融合：融合多个检测结果的位置信息
Eigen::Vector3d DataFusion::fusePositions(const std::vector<Detection>& detections, Eigen::Matrix3d& covariance) {
    if (detections.empty()) {
        covariance = Eigen::Matrix3d::Zero();
        return Eigen::Vector3d::Zero();
    }

    // 使用加权最小二乘融合，权重为协方差的逆
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    Eigen::Vector3d weighted_sum = Eigen::Vector3d::Zero();

    for (const auto& det : detections) {
        // 获取传感器校准的协方差矩阵
        Eigen::Matrix3d cov = det.covariance;

        // 计算协方差的逆作为权重（添加小的对角项防止奇异矩阵）
        Eigen::Matrix3d inv_cov = (cov + Eigen::Matrix3d::Identity() * 1e-6).inverse();

        W += inv_cov;
        weighted_sum += inv_cov * det.position_global;
    }

    // 计算融合位置
    Eigen::Vector3d fused_pos = W.inverse() * weighted_sum;

    // 融合协方差
    covariance = W.inverse();

    return fused_pos;
}

// 速度融合：融合多个检测结果的速度信息
Eigen::Vector3d DataFusion::fuseVelocities(const std::vector<Detection>& detections, Eigen::Matrix3d& covariance) {
    if (detections.empty()) {
        covariance = Eigen::Matrix3d::Zero();
        return Eigen::Vector3d::Zero();
    }

    // 与位置融合类似，但使用速度信息
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    Eigen::Vector3d weighted_sum = Eigen::Vector3d::Zero();

    for (const auto& det : detections) {
        // 速度协方差通常设为位置协方差的一半
        Eigen::Matrix3d cov = det.covariance * 0.5;

        // 计算协方差的逆作为权重
        Eigen::Matrix3d inv_cov = (cov + Eigen::Matrix3d::Identity() * 1e-6).inverse();

        W += inv_cov;
        weighted_sum += inv_cov * det.velocity_global;
    }

    // 计算融合速度
    Eigen::Vector3d fused_vel = W.inverse() * weighted_sum;

    // 融合协方差
    covariance = W.inverse();

    return fused_vel;
}
