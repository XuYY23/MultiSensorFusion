#include "MathUtils.h"

// 计算3维速度向量的余弦相似度
double MathUtils::cosineSimilarity(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
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
double MathUtils::cosineSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
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

double MathUtils::gaussianSimilarity(const double& dis, const double& scale) {
    if (scale <= 0) {
        return 0.0;
    }
    double sim = exp(-0.5 * pow(dis / scale, 2));
    return sim;
}

double MathUtils::calculateKL(const FeatureVector& a, const FeatureVector& b) {
    // 选择非空特征（优先视觉特征）
    const std::vector<double>* feat_a = &a.visual_features;
    const std::vector<double>* feat_b = &b.visual_features;
    if (feat_a->empty() || feat_b->empty()) {
        feat_a = &a.radar_features;
        feat_b = &b.radar_features;
    }
    if (feat_a->empty() || feat_b->empty() || feat_a->size() != feat_b->size()) {
        return 0.0;
    }

    const double eps = 1e-10;
    double kl = 0.0;
    // 归一化特征为概率分布
    double sum_a = std::accumulate(feat_a->begin(), feat_a->end(), 0.0);
    double sum_b = std::accumulate(feat_b->begin(), feat_b->end(), 0.0);
    sum_a = sum_a < eps ? 1.0 : sum_a;
    sum_b = sum_b < eps ? 1.0 : sum_b;

    for (size_t i = 0; i < feat_a->size(); ++i) {
        double p = std::max((*feat_a)[i] / sum_a, eps);
        double q = std::max((*feat_b)[i] / sum_b, eps);
        kl += p * std::log(p / q);
    }
    return kl;
}
