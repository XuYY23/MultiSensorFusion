#include "MultiTargetAssociator.h"
#include "Config.h"
#include "MathUtils.h"
#include "ReuseFunction.h"

MultiTargetAssociator::MultiTargetAssociator() {
    // 初始化目标ID计数器
    next_global_id_ = 1;
}

MultiTargetAssociator::~MultiTargetAssociator() {}

// 关联新检测结果与已有目标
std::vector<FusedObject> MultiTargetAssociator::associateTargets(const std::vector<Detection>& new_detections, const std::vector<FusedObject>& existing_targets, Timestamp current_time) {

    std::vector<FusedObject> updated_targets = existing_targets;

    // 如果没有新检测结果，直接返回现有目标
    if (new_detections.empty()) {
        return updated_targets;
    }

    // 如果没有现有目标，将所有新检测结果作为新目标
    if (existing_targets.empty()) {
        for (const Detection& det : new_detections) {
            FusedObject new_object;
            new_object.global_id = next_global_id_++;
            new_object.timestamp = current_time;
            new_object.position = det.position_global;
            new_object.velocity = det.velocity_global;
            new_object.position_covariance = Eigen::Matrix3d::Identity() * 0.1;
            new_object.velocity_covariance = Eigen::Matrix3d::Identity() * 0.05;
            new_object.class_probabilities[det.detected_class] = det.class_confidence;
            new_object.fused_features = det.features;
            new_object.associated_detections.push_back(det);
            new_object.track_length = 1;
            new_object.is_new = true;

            updated_targets.push_back(new_object);
        }
        return updated_targets;
    }

    // 1. 预测现有目标到当前时间的状态
    std::vector<FusedObject> predicted_targets;
    for (const auto& target : existing_targets) {
        predicted_targets.push_back(motion_model_.predict(target, target.timestamp, current_time));
    }

    // 2. 构建成本矩阵（成本 = 1 - 相似度）
    std::vector<std::vector<double>> cost_matrix(
        predicted_targets.size(),
        std::vector<double>(new_detections.size(), 1.0)  // 初始成本设为最大值
    );

    for (size_t i = 0; i < predicted_targets.size(); ++i) {
        // 创建一个虚拟检测用于相似度计算
        Detection predicted_det;
        predicted_det.position_global = predicted_targets[i].position;
        predicted_det.velocity_global = predicted_targets[i].velocity;
        /*predicted_det.detected_class =
            std::max_element(predicted_targets[i].class_probabilities.begin(),
                             predicted_targets[i].class_probabilities.end(),
                                [](const auto& a, const auto& b) {
                                    return a.second < b.second;
                                }
                            )->first;*/
		predicted_det.detected_class = predicted_targets[i].final_class;
        predicted_det.features = predicted_targets[i].fused_features;

        for (size_t j = 0; j < new_detections.size(); ++j) {
            double similarity = ReuseFunction::GetInstance().calculateSimilarity(predicted_det, new_detections[j]);
            cost_matrix[i][j] = 1.0 - similarity;  // 转换为成本
        }
    }

    // 3. 使用匈牙利算法找到最优关联
    std::vector<std::pair<int, int>> associations = ReuseFunction::GetInstance().hungarianAlgorithm(cost_matrix);

    // 4. 标记已关联的检测和目标
    std::vector<bool> detection_used(new_detections.size(), false);
    std::vector<bool> target_used(predicted_targets.size(), false);

    for (const auto& [target_idx, det_idx] : associations) {
        if (cost_matrix[target_idx][det_idx] < (1.0 - Config::GetInstance().getMinSimilarityThreshold()) && target_used[target_idx] == false) {
            // 更新目标状态
            FusedObject updated = motion_model_.update(
                predicted_targets[target_idx],
                new_detections[det_idx]
            );

            // 更新其他属性
            updated.global_id = existing_targets[target_idx].global_id;
            updated.track_length = existing_targets[target_idx].track_length + 1;
            updated.is_new = false;
            updated.associated_detections.push_back(new_detections[det_idx]);

            updated_targets[target_idx] = updated;

            detection_used[det_idx] = true;
            target_used[target_idx] = true;
        }
    }

    // 5. 处理未关联的检测结果（新目标）
    for (size_t j = 0; j < new_detections.size(); ++j) {
        if (!detection_used[j]) {
            FusedObject new_object;
            new_object.global_id = next_global_id_++;
            new_object.timestamp = current_time;
            new_object.position = new_detections[j].position_global;
            new_object.velocity = new_detections[j].velocity_global;
            new_object.position_covariance = Eigen::Matrix3d::Identity() * 0.1;
            new_object.velocity_covariance = Eigen::Matrix3d::Identity() * 0.05;
            new_object.class_probabilities[new_detections[j].detected_class] = new_detections[j].class_confidence;
            new_object.fused_features = new_detections[j].features;
            new_object.associated_detections.push_back(new_detections[j]);
            new_object.track_length = 1;
            new_object.is_new = true;

            updated_targets.push_back(new_object);
        }
    }

    return updated_targets;
}

std::vector<FusedObject> MultiTargetAssociator::associateTargets(const std::vector<Detection>& detections, Timestamp current_time) {
	std::vector<FusedObject> fused_objects;
    for(const Detection& det : detections) {
        if (det.detection_confidence < Config::GetInstance().getConfThreshold()) {
			continue; // 跳过低置信度检测
        }

		int best_match_idx = -1;
        double max_similarity = std::numeric_limits<double>::min();
        auto toString = [](const Eigen::Vector3d& v) -> std::string {
            return "(" + std::to_string(v.x()) + ", " + std::to_string(v.y()) + ", " + std::to_string(v.z()) + ")";
        };
		// 找到与当前检测结果最相似的融合目标
        for (int i = 0; i < fused_objects.size(); ++i) {
            double pos_sim = MathUtils::GetInstance().gaussianSimilarity(
                (det.position_global - fused_objects[i].position).norm(),
                Config::GetInstance().getMaxPositionDistance()
			);
            double vel_sim = MathUtils::GetInstance().cosineSimilarity(det.velocity_global, fused_objects[i].velocity);
            double similarity = Config::GetInstance().getPositionWeight() * pos_sim
				                + Config::GetInstance().getVelocityWeight() * vel_sim;
            if (similarity > max_similarity && similarity > Config::GetInstance().getMinSimilarityThreshold()) {
                max_similarity = similarity;
				best_match_idx = i;
            }
        }

        if (best_match_idx == -1) {
            FusedObject new_object;
            new_object.global_id = next_global_id_++;
            new_object.timestamp = current_time;
            // 这里的位置和速度只是需要和后面的目标进行目标匹配，后续还会进行融合
            new_object.position = det.position_global;
            new_object.velocity = det.velocity_global;
            new_object.associated_detections.push_back(det);
            new_object.is_new = true;
            fused_objects.push_back(new_object);
            std::cout << "新目标：new_object.position = " << toString(det.position_global) << ", new_object.velocity = " << toString(det.velocity_global) << std::endl;
        } else {
            std::cout << "目标匹配成功：Det.position_global = " << toString(det.position_global) << ", Det.velocity_global = " << toString(det.velocity_global)
                << ", fused_objects[best_match_idx].position = " << toString(fused_objects[best_match_idx].position) << ", fused_objects[best_match_idx].velocity = " << toString(fused_objects[best_match_idx].velocity)
                << std::endl;
			fused_objects[best_match_idx].associated_detections.push_back(det);
        }
	}

    return fused_objects;
}