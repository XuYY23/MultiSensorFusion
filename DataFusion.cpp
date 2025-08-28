#include "DataFusion.h"

// �������ںϣ��ں϶�������������
FeatureVector DataFusion::fuseFeatures(const std::vector<Detection>& detections) {
    FeatureVector fused;
    if (detections.empty()) {
        return fused;
    }

    // ��ÿ����������ƽ���ں�
    std::vector<std::vector<double>> all_visual, all_radar, all_audio, all_shape, all_motion;

    // �ռ���������
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

    // �����Ӿ�����ƽ��ֵ
    if (!all_visual.empty()) {
        size_t dim = all_visual[0].size();
        fused.visual_features.resize(dim, 0.0);
        for (const auto& feat : all_visual) {
            for (size_t i = 0; i < dim && i < feat.size(); ++i) {
                fused.visual_features[i] += feat[i] / all_visual.size();
            }
        }
    }

    // �����״�����ƽ��ֵ
    if (!all_radar.empty()) {
        size_t dim = all_radar[0].size();
        fused.radar_features.resize(dim, 0.0);
        for (const auto& feat : all_radar) {
            for (size_t i = 0; i < dim && i < feat.size(); ++i) {
                fused.radar_features[i] += feat[i] / all_radar.size();
            }
        }
    }

    // ������Ƶ����ƽ��ֵ
    if (!all_audio.empty()) {
        size_t dim = all_audio[0].size();
        fused.audio_features.resize(dim, 0.0);
        for (const auto& feat : all_audio) {
            for (size_t i = 0; i < dim && i < feat.size(); ++i) {
                fused.audio_features[i] += feat[i] / all_audio.size();
            }
        }
    }

    // ������״����ƽ��ֵ
    if (!all_shape.empty()) {
        size_t dim = all_shape[0].size();
        fused.shape_features.resize(dim, 0.0);
        for (const auto& feat : all_shape) {
            for (size_t i = 0; i < dim && i < feat.size(); ++i) {
                fused.shape_features[i] += feat[i] / all_shape.size();
            }
        }
    }

    // �����˶�����ƽ��ֵ
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

// ���߼��ںϣ��ں϶�������������жϣ�����D-S֤�����ۣ�
std::map<ObjectClass, double> DataFusion::fuseDecisions(const std::vector<Detection>& detections, double conf_threshold, bool& has_category_conflict) {
    std::map<ObjectClass, double> fused;
    if (detections.empty()) {
        return fused;
    }

    // ��ʼ���������ʷ���
    std::vector<std::map<ObjectClass, double>> bpas;
	std::set<ObjectClass> clases;
    for (const auto& det : detections) {
        if (det.detection_confidence < conf_threshold) {
            continue;  // ������Ŷȵ�����ֵ�������ü��
        }
        std::map<ObjectClass, double> bpa;
        // ���ڼ�⵽����𣬷��������Ŷ���صĸ���
        double confidence = det.class_confidence;
        bpa[det.detected_class] = confidence * 0.9;  // 90%�����������
        bpa[ObjectClass::UNKNOWN] = 1.0 - confidence * 0.9;  // ʣ�ಿ��Ϊ��ȷ��
        if(det.detected_class != ObjectClass::UNKNOWN) {
            clases.insert(det.detected_class);
		}
        bpas.push_back(bpa);
    }

    if(clases.size() > 1 || clases.size() == 0) {
        has_category_conflict = true;  // ���ڶ������û�п��ŵļ���������Ϊ��ͻ
		return std::map<ObjectClass, double>();
    } else {
        has_category_conflict = false; // ֻ��һ�����û�г�ͻ
	}

    // ���ֻ��һ���������ֱ�ӷ�����BPA
    if (bpas.size() == 1) {
        return bpas[0];
    }

    // Dempster��Ϲ����ںϵ�һ���͵ڶ���BPA
    std::map<ObjectClass, double> combined = bpas[0];
    for (size_t i = 1; i < bpas.size(); ++i) {
        std::map<ObjectClass, double> temp;
        double conflict = 0.0;
        
        // �������п��ܵĽ���
        for (const auto& [a, mass_a] : combined) {
            for (const auto& [b, mass_b] : bpas[i]) {
                if (a == b) {
                    // ͬһ���ֱ�����
                    temp[a] += mass_a * mass_b;
                } else if (a != ObjectClass::UNKNOWN && b != ObjectClass::UNKNOWN) {
                    // ��ͬ������ڳ�ͻ
                    conflict += mass_a * mass_b;
                } else {
                    // һ����UNKNOWN�����Ϊ��UNKNOWN�����
                    ObjectClass non_unknown = (a != ObjectClass::UNKNOWN) ? a : b;
                    temp[non_unknown] += mass_a * mass_b;
                }
            }
        }

        // ��һ��������1 - conflict��
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

// λ���ںϣ��ں϶���������λ����Ϣ
Eigen::Vector3d DataFusion::fusePositions(const std::vector<Detection>& detections, Eigen::Matrix3d& covariance) {
    if (detections.empty()) {
        covariance = Eigen::Matrix3d::Zero();
        return Eigen::Vector3d::Zero();
    }

    // ʹ�ü�Ȩ��С�����ںϣ�Ȩ��ΪЭ�������
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    Eigen::Vector3d weighted_sum = Eigen::Vector3d::Zero();

    for (const auto& det : detections) {
        // ��ȡ������У׼��Э�������
        Eigen::Matrix3d cov = det.covariance;

        // ����Э���������ΪȨ�أ����С�ĶԽ����ֹ�������
        Eigen::Matrix3d inv_cov = (cov + Eigen::Matrix3d::Identity() * 1e-6).inverse();

        W += inv_cov;
        weighted_sum += inv_cov * det.position_global;
    }

    // �����ں�λ��
    Eigen::Vector3d fused_pos = W.inverse() * weighted_sum;

    // �ں�Э����
    covariance = W.inverse();

    return fused_pos;
}

// �ٶ��ںϣ��ں϶����������ٶ���Ϣ
Eigen::Vector3d DataFusion::fuseVelocities(const std::vector<Detection>& detections, Eigen::Matrix3d& covariance) {
    if (detections.empty()) {
        covariance = Eigen::Matrix3d::Zero();
        return Eigen::Vector3d::Zero();
    }

    // ��λ���ں����ƣ���ʹ���ٶ���Ϣ
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    Eigen::Vector3d weighted_sum = Eigen::Vector3d::Zero();

    for (const auto& det : detections) {
        // �ٶ�Э����ͨ����Ϊλ��Э�����һ��
        Eigen::Matrix3d cov = det.covariance * 0.5;

        // ����Э���������ΪȨ��
        Eigen::Matrix3d inv_cov = (cov + Eigen::Matrix3d::Identity() * 1e-6).inverse();

        W += inv_cov;
        weighted_sum += inv_cov * det.velocity_global;
    }

    // �����ں��ٶ�
    Eigen::Vector3d fused_vel = W.inverse() * weighted_sum;

    // �ں�Э����
    covariance = W.inverse();

    return fused_vel;
}
