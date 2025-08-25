#include "ReuseFunction.h"
#include "MathUtils.h"
#include "Config.h"

// 计算两个检测结果的相似度
double ReuseFunction::calculateSimilarity(const Detection& a, const Detection& b) {
    // 1. 位置相似度 (使用高斯核)
    double pos_dist = (a.position_global - b.position_global).norm();
    double pos_sim = MathUtils::GetInstance().gaussianSimilarity(pos_dist, Config::GetInstance().getMaxPositionDistance());

    // 2. 速度相似度
    double vel_diff = (a.velocity_global - b.velocity_global).norm();
    double vel_sim = MathUtils::GetInstance().gaussianSimilarity(vel_diff, Config::GetInstance().getMaxVelocityDiff());

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
double ReuseFunction::calculateFeatureSimilarity(const FeatureVector& a, const FeatureVector& b) {
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
        total_sim += MathUtils::GetInstance().cosineSimilarity(a.visual_features, b.visual_features);
        feature_count++;
    }

    // 雷达特征相似度
    if (!a.radar_features.empty() && !b.radar_features.empty()) {
        total_sim += MathUtils::GetInstance().cosineSimilarity(a.radar_features, b.radar_features);
        feature_count++;
    }

    // 音频特征相似度
    if (!a.audio_features.empty() && !b.audio_features.empty()) {
        total_sim += MathUtils::GetInstance().cosineSimilarity(a.audio_features, b.audio_features);
        feature_count++;
    }

    // 形状特征相似度
    if (!a.shape_features.empty() && !b.shape_features.empty()) {
        total_sim += MathUtils::GetInstance().cosineSimilarity(a.shape_features, b.shape_features);
        feature_count++;
    }

    // 运动特征相似度
    if (!a.motion_features.empty() && !b.motion_features.empty()) {
        total_sim += MathUtils::GetInstance().cosineSimilarity(a.motion_features, b.motion_features);
        feature_count++;
    }

    // 计算平均相似度
    return feature_count > 0 ? total_sim / feature_count : 0.0;
}

// 使用dlib库实现匈牙利算法（20.0版本）
std::vector<std::pair<int, int>> ReuseFunction::hungarianAlgorithm(const std::vector<std::vector<double>>& cost_matrix) {
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
            }
            else {
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

std::vector<double> ReuseFunction::calculateLocalDensity(const std::vector<FeatureVector>& features, double d_c) {
    std::vector<double> rho(features.size(), 0.0);
    for (size_t i = 0; i < features.size(); ++i) {
        for (size_t j = 0; j < features.size(); ++j) {
            if (i == j) {
                continue;
            }

            // 距离=1-余弦相似度
            double dist = 1 - calculateFeatureSimilarity(features[i], features[j]);
            if (dist < d_c) {
                rho[i] += 1.0; // 指示函数χ(d_ij - d_c)
            }
        }
    }
    return rho;
}

std::vector<double> ReuseFunction::calculateRelativeDistance(const std::vector<FeatureVector>& features, const std::vector<double>& rho) {
    std::vector<double> delta(features.size(), 0.0);
    double max_rho = *std::max_element(rho.begin(), rho.end());

    for (size_t i = 0; i < features.size(); ++i) {
        if (rho[i] == max_rho) {
            // 全局密度最大，δ取最大距离
            double max_dist = 0.0;
            for (size_t j = 0; j < features.size(); ++j) {
                if (i == j) {
                    continue;
                }
                double dist = 1 - calculateFeatureSimilarity(features[i], features[j]);
                max_dist = std::max(max_dist, dist);
            }
            delta[i] = max_dist;
        } else {
            // 找ρ>当前样本的最小距离
            double min_dist = 1e9;
            for (size_t j = 0; j < features.size(); ++j) {
                if (rho[j] <= rho[i]) {
                    continue;
                }
                double dist = 1 - calculateFeatureSimilarity(features[i], features[j]);
                min_dist = std::min(min_dist, dist);
            }
            delta[i] = min_dist;
        }
    }
    return delta;
}

void ReuseFunction::generatePseudoLabels(std::vector<ClusterResult>& clusters,
                                         const std::map<std::shared_ptr<BaseObject>, std::vector<FeatureVector>>& historical_features,
                                         const std::vector<std::map<std::string, std::string>>& meta_datas) {
    int idx = 0;
    for (ClusterResult& cluster : clusters) {
        PseudoLabel label;
        label.new_class = cluster.cluster_class;

        // 找关联的历史类别（Top1相似度）
        double max_sim = 0.0;
        std::shared_ptr<BaseObject> associated_cls = std::make_shared<BaseObject>();
        for (const auto& [cls, hist_feats] : historical_features) {
            if (hist_feats.empty()) {
                continue;
            }
            // 计算簇中心与历史特征的平均相似度
            double avg_sim = 0.0;
            for (const auto& feat : hist_feats) {
                avg_sim += calculateFeatureSimilarity(cluster.cluster_center, feat);
            }
            avg_sim /= hist_feats.size();
            if (avg_sim > max_sim) {
                max_sim = avg_sim;
                associated_cls = cls;
            }
        }
        label.associated_historical_class = associated_cls;
        if(idx >= meta_datas.size()) {
            std::cerr << "Warning: Metadata size is less than clusters size." << std::endl;
            label.metadata = {};
        } else {
            label.metadata = meta_datas[idx];
        }
		
        // 组装伪标签字符串
        label.label = "新目标-" + label.new_class->toString() + "-近似类别" + label.associated_historical_class->toString();
        for (const auto& [key, value] : label.metadata) {
			label.label += "-" + value;
        }
        cluster.pseudo_label = label;
        ++idx;
    }
}
