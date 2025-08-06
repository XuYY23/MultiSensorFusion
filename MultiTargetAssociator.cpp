#include "MultiTargetAssociator.h"
#include "FeatureExtractor.h"

MultiTargetAssociator::MultiTargetAssociator() {
    // ��ʼ��Ŀ��ID������
    next_global_id_ = 1;

    // ���ù�������
    position_weight_ = 0.4;             // λ��Ȩ��
    velocity_weight_ = 0.2;             // �ٶ�Ȩ��
    class_weight_ = 0.2;                // ���Ȩ��
    feature_weight_ = 0.2;              // ����Ȩ��
    max_position_distance_ = 10.0;      // ���λ�þ�����ֵ(��)
    max_velocity_diff_ = 5.0;           // ����ٶȲ���ֵ(��/��)
    min_similarity_threshold_ = 0.5;    // ��С���ƶ���ֵ
}

MultiTargetAssociator::~MultiTargetAssociator() {}

// �����¼����������Ŀ��
std::vector<FusedObject> MultiTargetAssociator::associateTargets(const std::vector<Detection>& new_detections, const std::vector<FusedObject>& existing_targets, Timestamp current_time) {

    std::vector<FusedObject> updated_targets = existing_targets;

    // ���û���¼������ֱ�ӷ�������Ŀ��
    if (new_detections.empty()) {
        return updated_targets;
    }

    // ���û������Ŀ�꣬�������¼������Ϊ��Ŀ��
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

    // 1. Ԥ������Ŀ�굽��ǰʱ���״̬
    std::vector<FusedObject> predicted_targets;
    for (const auto& target : existing_targets) {
        predicted_targets.push_back(motion_model_.predict(target, target.timestamp, current_time));
    }

    // 2. �����ɱ����󣨳ɱ� = 1 - ���ƶȣ�
    std::vector<std::vector<double>> cost_matrix(
        predicted_targets.size(),
        std::vector<double>(new_detections.size(), 1.0)  // ��ʼ�ɱ���Ϊ���ֵ
    );

    for (size_t i = 0; i < predicted_targets.size(); ++i) {
        // ����һ���������������ƶȼ���
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
            cost_matrix[i][j] = 1.0 - similarity;  // ת��Ϊ�ɱ�
        }
    }

    // 3. ʹ���������㷨�ҵ����Ź���
    std::vector<std::pair<int, int>> associations = hungarianAlgorithm(cost_matrix);

    // 4. ����ѹ����ļ���Ŀ��
    std::vector<bool> detection_used(new_detections.size(), false);
    std::vector<bool> target_used(predicted_targets.size(), false);

    for (const auto& [target_idx, det_idx] : associations) {
        if (cost_matrix[target_idx][det_idx] < (1.0 - min_similarity_threshold_) && target_used[target_idx] == false) {
            // ����Ŀ��״̬
            FusedObject updated = motion_model_.update(
                predicted_targets[target_idx],
                new_detections[det_idx]
            );

            // ������������
            updated.global_id = existing_targets[target_idx].global_id;
            updated.track_length = existing_targets[target_idx].track_length + 1;
            updated.is_new = false;
            updated.associated_detections.push_back(new_detections[det_idx]);

            updated_targets[target_idx] = updated;

            detection_used[det_idx] = true;
            target_used[target_idx] = true;
        }
    }

    // 5. ����δ�����ļ��������Ŀ�꣩
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

// ������������������ƶ�
double MultiTargetAssociator::calculateSimilarity(const Detection& a, const Detection& b) {
    // 1. λ�����ƶ� (ʹ�ø�˹��)
    double pos_dist = (a.position_global - b.position_global).norm();
    double pos_sim = exp(-0.5 * pow(pos_dist / max_position_distance_, 2));

    // 2. �ٶ����ƶ�
    double vel_diff = (a.velocity_global - b.velocity_global).norm();
    double vel_sim = exp(-0.5 * pow(vel_diff / max_velocity_diff_, 2));

    // 3. ������ƶ�
    double class_sim = (a.detected_class == b.detected_class) ? 1.0 : 0.0;
    // ����������Ŷ�
    class_sim *= (a.class_confidence + b.class_confidence) * 0.5;

    // 4. �������ƶ� (ʹ���������ƶ�)
    double feat_sim = calculateFeatureSimilarity(a.features, b.features);

    // ��Ȩ�ں����ƶ�
    double similarity = position_weight_ * pos_sim 
                        + velocity_weight_ * vel_sim 
                        + class_weight_ * class_sim 
                        + feature_weight_ * feat_sim;

    return similarity;
}

// �����������������ƶ�
double MultiTargetAssociator::calculateFeatureSimilarity(const FeatureVector& a, const FeatureVector& b) {
    // �����������������Ϊ�գ����ƶ�Ϊ1.0
    if (a.isEmpty() && b.isEmpty()) {
        return 1.0;
    }

    // ���һ��Ϊ�գ�һ����Ϊ�գ����ƶ�Ϊ0.0
    if (a.isEmpty() || b.isEmpty()) {
        return 0.0;
    }

    double total_sim = 0.0;
    int feature_count = 0;

    // �Ӿ��������ƶ�
    if (!a.visual_features.empty() && !b.visual_features.empty()) {
        total_sim += cosineSimilarity(a.visual_features, b.visual_features);
        feature_count++;
    }

    // �״��������ƶ�
    if (!a.radar_features.empty() && !b.radar_features.empty()) {
        total_sim += cosineSimilarity(a.radar_features, b.radar_features);
        feature_count++;
    }

    // ��Ƶ�������ƶ�
    if (!a.audio_features.empty() && !b.audio_features.empty()) {
        total_sim += cosineSimilarity(a.audio_features, b.audio_features);
        feature_count++;
    }

    // ��״�������ƶ�
    if (!a.shape_features.empty() && !b.shape_features.empty()) {
        total_sim += cosineSimilarity(a.shape_features, b.shape_features);
        feature_count++;
    }

    // �˶��������ƶ�
    if (!a.motion_features.empty() && !b.motion_features.empty()) {
        total_sim += cosineSimilarity(a.motion_features, b.motion_features);
        feature_count++;
    }

    // ����ƽ�����ƶ�
    return feature_count > 0 ? total_sim / feature_count : 0.0;
}

// ���������������������ƶ�
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

// ʹ��dlib��ʵ���������㷨��20.0�汾��
std::vector<std::pair<int, int>> MultiTargetAssociator::hungarianAlgorithm(const std::vector<std::vector<double>>& cost_matrix) {
    std::vector<std::pair<int, int>> result;

    if (cost_matrix.empty() || cost_matrix[0].empty()) {
        return result;
    }

    // ��ȡ�����С
    int rows = cost_matrix.size();
    int cols = cost_matrix[0].size();

    // dlib�����ɱ������㷨��Ҫ����������Ƿ��������
    int n = std::max(rows, cols);

    // �����ɱ�����dlibʹ��������⣬����������һ����ֵ��ȥ�ɱ���
    dlib::matrix<int> assignment_matrix(n, n);
    const int MAX_COST = 1000000;  // �㹻��ĳ���

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i < rows && j < cols) {
                // ת��Ϊ��������ת�ɱ�����Ϊdlibʵ�ֵ������ɱ����䣩
                assignment_matrix(i, j) = static_cast<int>(MAX_COST - cost_matrix[i][j] * MAX_COST);
            } else {
                // ����λ�óɱ���Ϊ0
                assignment_matrix(i, j) = 0;
            }
        }
    }

    // ִ�����ɱ������㷨
    std::vector<long> assignment = max_cost_assignment(assignment_matrix);

    // ��������ֻ������Чƥ��
    for (int i = 0; i < rows; ++i) {
        if (assignment[i] < cols) {  // ֻ��������Ч��Χ�ڵ�ƥ��
            // ���ƥ���Ƿ���Ч���ɱ�������ֵ��
            if (cost_matrix[i][assignment[i]] < (1.0 - min_similarity_threshold_)) {
                result.emplace_back(i, assignment[i]);
            }
        }
    }

    return result;
}