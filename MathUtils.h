#pragma once

#include "structs.h"

class MathUtils {
	void init() {}
	INSTANCE(MathUtils)
public:
	// 计算两个向量的余弦相似度
	double cosineSimilarity(const Eigen::Vector3d& a, const Eigen::Vector3d& b);
	double cosineSimilarity(const std::vector<double>& a, const std::vector<double>& b);

	// 计算高斯相似度（基于位置和速度的马氏距离）
	double gaussianSimilarity(const double& dis, const double& scale);

	// 计算KL散度（特征的差异程度，该值越小说明两个特征越相似）
	double calculateKL(const FeatureVector& a, const FeatureVector& b);
};