#include "MultimodalDetectorSystem.h"
#include "Config.h"
#include "FeatureDataBase.h"
#include "ReuseFunction.h"
#include "MathUtils.h"
#include "IncrementalLearning.h"

MultimodalDetectorSystem::MultimodalDetectorSystem(const std::map<std::string, SensorCalibration>& calibrations) : 
    aligner_(calibrations), 
    use_custom_target_time_(false) {

    // 分离一个增量学习线程，每过N秒学习一次
    std::thread(
        [&]() {
            while (true) {
                if (FeatureDataBase::GetInstance().getNewCategories().size() >= Config::GetInstance().getNumGap()) {
                    IncrementalLearning::GetInstance().startIncrementalTraining();
                }
				std::this_thread::sleep_for(std::chrono::seconds(Config::GetInstance().getTimeGap())); // 每N秒进行一次增量学习
            }
        }
    ).detach();

}

// 添加新的检测结果
void MultimodalDetectorSystem::addDetections(const std::vector<Detection>& detections) {
    // 对每个检测结果进行时间对齐，并添加到队列
    for (const auto& det : detections) {
        Detection aligned = det;
        aligner_.alignTemporal(aligned);  // 先进行时间对齐
        current_detections_.push_back(aligned);
    }
}

// 处理当前所有检测结果
void MultimodalDetectorSystem::processDetections() {
    if (current_detections_.empty()) {
        return;
    }

    // 确定目标时间戳
    Timestamp target_time;
    if (use_custom_target_time_) {
        target_time = target_timestamp_;
    } else {
        // 使用最新检测结果的时间戳
        target_time = current_detections_[0].timestamp;
        for (const auto& det : current_detections_) {
            if (det.timestamp > target_time) {
                target_time = det.timestamp;
            }
        }
    }

    // 1. 时间同步：将所有检测结果插值到目标时间点
    std::vector<Detection> synchronized_detections = aligner_.synchronizeDetections(current_detections_, target_time);

    // 2. 多目标关联
    fused_objects_ = associator_.associateTargets(synchronized_detections, target_time);

    // 3. 对每个目标进行特征和决策融合
    for (auto& obj : fused_objects_) {
        if (!obj.associated_detections.empty()) {
            // 决策级融合
            obj.class_probabilities = fusion_.fuseDecisions(obj.associated_detections, Config::GetInstance().getConfThreshold(), obj.has_category_conflict);

            if (obj.has_category_conflict == true) {
                // 这里应该重新检测
				continue;  // 如果存在类别冲突，则跳过该目标
            }

            // 特征级融合
            obj.fused_features = fusion_.fuseFeatures(obj.associated_detections);

            // 如果是新目标，需要融合位置和速度
            //if (obj.is_new) {
                // 直接融合位置和速度，因为目标关联不再是目标跟踪，永远是新目标
                Eigen::Matrix3d pos_cov, vel_cov;
                obj.position = fusion_.fusePositions(obj.associated_detections, pos_cov);
                obj.velocity = fusion_.fuseVelocities(obj.associated_detections, vel_cov);
                obj.position_covariance = pos_cov;
                obj.velocity_covariance = vel_cov;
            //}
        }
    }

    // 4. 确定最终类别
    for (auto& obj : fused_objects_) {
        if (obj.class_probabilities.empty()) {
            obj.final_class = ObjectClass::UNKNOWN;
            obj.final_class_confidence = 0.0;
            continue;
        }

        // 找到概率最高的类别
        auto max_it = std::max_element(
            obj.class_probabilities.begin(),
            obj.class_probabilities.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        );

        // 设定置信度阈值（如0.5），低于阈值则标记为UNKNOWN
        const double CONFIDENCE_THRESHOLD = 0.5;
        if (max_it->second >= CONFIDENCE_THRESHOLD) {
            obj.final_class = max_it->first;
            obj.final_class_confidence = max_it->second;
        } else {
            obj.final_class = ObjectClass::UNKNOWN;
            obj.final_class_confidence = max_it->second;
        }
    }

    // 5、聚类
    clustering();

    // 清空当前检测队列
    current_detections_.clear();
}

// 获取当前融合结果
const std::vector<FusedObject>& MultimodalDetectorSystem::getFusedObjects() const {
    return fused_objects_;
}

// 保存融合结果到JSON文件
bool MultimodalDetectorSystem::saveResults(const std::string& filename) {
    return json_output_.saveResults(fused_objects_, filename);
}

// 设置时间同步的目标时间
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
        bool is_new = true; // 初始标记为新目标

        // 与所有历史特征对比
        for (const auto& [cls, hist_feats] : FeatureDataBase::GetInstance().getHistoricalFeatures()) {
            for (const auto& hist_feat : hist_feats) {
                if (hist_feat.isEmpty()) {
                    continue;
                }

                // 余弦相似度
                double cos_sim = ReuseFunction::GetInstance().calculateFeatureSimilarity(feat, hist_feat);
                // KL散度
                double kl_div = MathUtils::GetInstance().calculateKL(feat, hist_feat);

                // 若满足任一条件，判定为已知目标
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

    // 判定是否需要聚类（新样本数>30）
    int cluster_thresh = Config::GetInstance().getNewSampleClusterThreshold();
    return (FeatureDataBase::GetInstance().getNewFeatures().size() > cluster_thresh);
}
