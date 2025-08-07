#include "MultiTargetAssociator.h"
#include "FeatureExtractor.h"
#include "Config.h"

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
            double similarity = calculateSimilarity(predicted_det, new_detections[j]);
            cost_matrix[i][j] = 1.0 - similarity;  // 转换为成本
        }
    }

    // 3. 使用匈牙利算法找到最优关联
    std::vector<std::pair<int, int>> associations = hungarianAlgorithm(cost_matrix);

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
            double pos_sim = gaussianSimilarity(
                (det.position_global - fused_objects[i].position).norm(),
                Config::GetInstance().getMaxPositionDistance()
			);
            double vel_sim = cosineSimilarity(det.velocity_global, fused_objects[i].velocity);
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

// 计算两个检测结果的相似度
double MultiTargetAssociator::calculateSimilarity(const Detection& a, const Detection& b) {
    // 1. 位置相似度 (使用高斯核)
    double pos_dist = (a.position_global - b.position_global).norm();
    double pos_sim = gaussianSimilarity(pos_dist, Config::GetInstance().getMaxPositionDistance());

    // 2. 速度相似度
    double vel_diff = (a.velocity_global - b.velocity_global).norm();
    double vel_sim = gaussianSimilarity(vel_diff, Config::GetInstance().getMaxVelocityDiff());

    // 3. 类别相似度
    double class_sim = (a.detected_class == b.detected_class) ? 1.0 : 0.0;
    // 考虑类别置信度
    class_sim *= (a.class_confidence + b.class_confidence) * 0.5;

    // 4. 特征相似度 (使用余弦相似度)
    double feat_sim = calculateFeatureSimilarity(a.features, b.features);

    // 加权融合相似度
    double similarity = Config::GetInstance().getPositionWeight() * pos_sim
                        + Config::GetInstance().getVelocityWeight() * vel_sim
                        + Config::GetInstance().getClassWeight() * class_sim
                        + Config::GetInstance().getFeatureWeight() * feat_sim;

    return similarity;
}

// 计算特征向量的相似度
double MultiTargetAssociator::calculateFeatureSimilarity(const FeatureVector& a, const FeatureVector& b) {
    // 如果两个特征向量都为空，相似度为1.0
    if (a.isEmpty() && b.isEmpty()) {
        return 1.0;
    }

    // 如果一个为空，一个不为空，相似度为0.0
    if (a.isEmpty() || b.isEmpty()) {
        return 0.0;
    }

    double total_sim = 0.0;
    int feature_count = 0;

    // 视觉特征相似度
    if (!a.visual_features.empty() && !b.visual_features.empty()) {
        total_sim += cosineSimilarity(a.visual_features, b.visual_features);
        feature_count++;
    }

    // 雷达特征相似度
    if (!a.radar_features.empty() && !b.radar_features.empty()) {
        total_sim += cosineSimilarity(a.radar_features, b.radar_features);
        feature_count++;
    }

    // 音频特征相似度
    if (!a.audio_features.empty() && !b.audio_features.empty()) {
        total_sim += cosineSimilarity(a.audio_features, b.audio_features);
        feature_count++;
    }

    // 形状特征相似度
    if (!a.shape_features.empty() && !b.shape_features.empty()) {
        total_sim += cosineSimilarity(a.shape_features, b.shape_features);
        feature_count++;
    }

    // 运动特征相似度
    if (!a.motion_features.empty() && !b.motion_features.empty()) {
        total_sim += cosineSimilarity(a.motion_features, b.motion_features);
        feature_count++;
    }

    // 计算平均相似度
    return feature_count > 0 ? total_sim / feature_count : 0.0;
}

// 计算3维速度向量的余弦相似度
double MultiTargetAssociator::cosineSimilarity(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    // 计算向量点积（3个维度分别相乘后求和）
    double dot_product = a.dot(b);  // 等价于 a.x()*b.x() + a.y()*b.y() + a.z()*b.z()

    // 计算向量模长的平方（各维度平方和）
    double norm_a_squared = a.squaredNorm();  // 等价于 a.x()² + a.y()² + a.z()²
    double norm_b_squared = b.squaredNorm();

    // 避免除以零（若任一向量模长为0，返回0表示无相似度）
    if (norm_a_squared == 0.0 || norm_b_squared == 0.0) {
        return 0.0;
    }

    // 余弦相似度公式：点积 / (模长乘积)
    return dot_product / (sqrt(norm_a_squared) * sqrt(norm_b_squared));
}

// 计算两个向量的余弦相似度
double MultiTargetAssociator::cosineSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 0.0;
    }

    double dot_product = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a == 0.0 || norm_b == 0.0) {
        return 0.0;
    }

    return dot_product / (sqrt(norm_a) * sqrt(norm_b));
}

double MultiTargetAssociator::gaussianSimilarity(const double& dis, const double& scale) {
    if (scale <= 0) {
        return 0.0;
    }
    double sim = exp(-0.5 * pow(dis / scale, 2));
	return sim;
}

// 使用dlib库实现匈牙利算法（20.0版本）
std::vector<std::pair<int, int>> MultiTargetAssociator::hungarianAlgorithm(const std::vector<std::vector<double>>& cost_matrix) {
    std::vector<std::pair<int, int>> result;

    if (cost_matrix.empty() || cost_matrix[0].empty()) {
        return result;
    }

    // 获取矩阵大小
    int rows = cost_matrix.size();
    int cols = cost_matrix[0].size();

    // dlib的最大成本分配算法需要方阵，如果不是方阵则填充
    int n = std::max(rows, cols);

    // 创建成本矩阵（dlib使用最大化问题，所以这里用一个大值减去成本）
    dlib::matrix<int> assignment_matrix(n, n);
    const int MAX_COST = 1000000;  // 足够大的常数

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i < rows && j < cols) {
                // 转换为整数并反转成本（因为dlib实现的是最大成本分配）
                assignment_matrix(i, j) = static_cast<int>(MAX_COST - cost_matrix[i][j] * MAX_COST);
            } else {
                // 填充的位置成本设为0
                assignment_matrix(i, j) = 0;
            }
        }
    }

    // 执行最大成本分配算法
    std::vector<long> assignment = max_cost_assignment(assignment_matrix);

    // 处理结果，只保留有效匹配
    for (int i = 0; i < rows; ++i) {
        if (assignment[i] < cols) {  // 只考虑在有效范围内的匹配
            // 检查匹配是否有效（成本低于阈值）
            if (cost_matrix[i][assignment[i]] < (1.0 - Config::GetInstance().getMinSimilarityThreshold())) {
                result.emplace_back(i, assignment[i]);
            }
        }
    }

    return result;
}