#include "MultimodalDetectorSystem.h"
#include "Config.h"
#include "FeatureDataBase.h"
#include "ReuseFunction.h"
#include "MathUtils.h"
#include "IncrementalLearning.h"

MultimodalDetectorSystem::MultimodalDetectorSystem(const std::map<std::string, SensorCalibration>& calibrations) : 
    aligner_(calibrations), 
    use_custom_target_time_(false) {

    // ����һ������ѧϰ�̣߳�ÿ��N��ѧϰһ��
    std::thread(
        [&]() {
            while (true) {
                if (FeatureDataBase::GetInstance().getNewCategories().size() >= Config::GetInstance().getNumGap()) {
                    IncrementalLearning::GetInstance().startIncrementalTraining();
                }
				std::this_thread::sleep_for(std::chrono::seconds(Config::GetInstance().getTimeGap())); // ÿN�����һ������ѧϰ
            }
        }
    ).detach();

}

// ����µļ����
void MultimodalDetectorSystem::addDetections(const std::vector<Detection>& detections) {
    // ��ÿ�����������ʱ����룬����ӵ�����
    for (const auto& det : detections) {
        Detection aligned = det;
        aligner_.alignTemporal(aligned);  // �Ƚ���ʱ�����
        current_detections_.push_back(aligned);
    }
}

// ����ǰ���м����
void MultimodalDetectorSystem::processDetections() {
    if (current_detections_.empty()) {
        return;
    }

    // ȷ��Ŀ��ʱ���
    Timestamp target_time;
    if (use_custom_target_time_) {
        target_time = target_timestamp_;
    } else {
        // ʹ�����¼������ʱ���
        target_time = current_detections_[0].timestamp;
        for (const auto& det : current_detections_) {
            if (det.timestamp > target_time) {
                target_time = det.timestamp;
            }
        }
    }

    // 1. ʱ��ͬ���������м������ֵ��Ŀ��ʱ���
    std::vector<Detection> synchronized_detections = aligner_.synchronizeDetections(current_detections_, target_time);

    // 2. ��Ŀ�����
    fused_objects_ = associator_.associateTargets(synchronized_detections, target_time);

    // 3. ��ÿ��Ŀ����������;����ں�
    for (auto& obj : fused_objects_) {
        if (!obj.associated_detections.empty()) {
            // ���߼��ں�
            obj.class_probabilities = fusion_.fuseDecisions(obj.associated_detections, Config::GetInstance().getConfThreshold(), obj.has_category_conflict);

            if (obj.has_category_conflict == true) {
                // ����Ӧ�����¼��
				continue;  // �����������ͻ����������Ŀ��
            }

            // �������ں�
            obj.fused_features = fusion_.fuseFeatures(obj.associated_detections);

            // �������Ŀ�꣬��Ҫ�ں�λ�ú��ٶ�
            //if (obj.is_new) {
                // ֱ���ں�λ�ú��ٶȣ���ΪĿ�����������Ŀ����٣���Զ����Ŀ��
                Eigen::Matrix3d pos_cov, vel_cov;
                obj.position = fusion_.fusePositions(obj.associated_detections, pos_cov);
                obj.velocity = fusion_.fuseVelocities(obj.associated_detections, vel_cov);
                obj.position_covariance = pos_cov;
                obj.velocity_covariance = vel_cov;
            //}
        }
    }

    // 4. ȷ���������
    for (auto& obj : fused_objects_) {
        if (obj.class_probabilities.empty()) {
            obj.final_class = ObjectClass::UNKNOWN;
            obj.final_class_confidence = 0.0;
            continue;
        }

        // �ҵ�������ߵ����
        auto max_it = std::max_element(
            obj.class_probabilities.begin(),
            obj.class_probabilities.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        );

        // �趨���Ŷ���ֵ����0.5����������ֵ����ΪUNKNOWN
        const double CONFIDENCE_THRESHOLD = 0.5;
        if (max_it->second >= CONFIDENCE_THRESHOLD) {
            obj.final_class = max_it->first;
            obj.final_class_confidence = max_it->second;
        } else {
            obj.final_class = ObjectClass::UNKNOWN;
            obj.final_class_confidence = max_it->second;
        }
    }

    // 5������
    clustering();

    // ��յ�ǰ������
    current_detections_.clear();
}

// ��ȡ��ǰ�ںϽ��
const std::vector<FusedObject>& MultimodalDetectorSystem::getFusedObjects() const {
    return fused_objects_;
}

// �����ںϽ����JSON�ļ�
bool MultimodalDetectorSystem::saveResults(const std::string& filename) {
    return json_output_.saveResults(fused_objects_, filename);
}

// ����ʱ��ͬ����Ŀ��ʱ��
void MultimodalDetectorSystem::setTargetTimestamp(Timestamp target_time) {
    target_timestamp_ = target_time;
    use_custom_target_time_ = true;
}

void MultimodalDetectorSystem::clustering() {
    bool need_clustering = twoValueJudge();
    if (need_clustering) {
        std::vector<ClusterResult> cluster_result = FeatureDataBase::GetInstance().performDPCClustering();
        std::vector<std::map<std::string, std::string>> meta_datas;
        ReuseFunction::GetInstance().generatePseudoLabels(cluster_result, FeatureDataBase::GetInstance().getHistoricalFeatures(), meta_datas);
        FeatureDataBase::GetInstance().setNewCategories(cluster_result);
    }
}

bool MultimodalDetectorSystem::twoValueJudge() {
    std::vector<FeatureVector> fused_features;
    for (const auto& obj : fused_objects_) {
        if (!obj.fused_features.isEmpty()) {
            fused_features.push_back(obj.fused_features);
        }
    }

    const double kl_thresh = Config::GetInstance().getKlDivThreshold();
    const double cos_thresh = Config::GetInstance().getNewTargetCosineThresh();
    for (const FeatureVector& feat : fused_features) {
        if (feat.isEmpty()) {
            continue;
        }
        bool is_new = true; // ��ʼ���Ϊ��Ŀ��

        // ��������ʷ�����Ա�
        for (const auto& [cls, hist_feats] : FeatureDataBase::GetInstance().getHistoricalFeatures()) {
            for (const auto& hist_feat : hist_feats) {
                if (hist_feat.isEmpty()) {
                    continue;
                }

                // �������ƶ�
                double cos_sim = ReuseFunction::GetInstance().calculateFeatureSimilarity(feat, hist_feat);
                // KLɢ��
                double kl_div = MathUtils::GetInstance().calculateKL(feat, hist_feat);

                // ��������һ�������ж�Ϊ��֪Ŀ��
                if (cos_sim >= cos_thresh || kl_div <= kl_thresh) {
                    is_new = false;
                    break;
                }
            }
            if (!is_new) {
                break;
            }
        }

        if (is_new) {
            FeatureDataBase::GetInstance().addNewFeature(feat);
        }
    }

    // �ж��Ƿ���Ҫ���ࣨ��������>30��
    int cluster_thresh = Config::GetInstance().getNewSampleClusterThreshold();
    return (FeatureDataBase::GetInstance().getNewFeatures().size() > cluster_thresh);
}
