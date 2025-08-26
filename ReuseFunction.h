#pragma once

#include "structs.h"

class ReuseFunction {
	void init() {}
	INSTANCE(ReuseFunction)

public:
	// 计算两个检测结果的相似度
	double calculateSimilarity(const Detection& a, const Detection& b);

	// 计算特征向量的相似度
	double calculateFeatureSimilarity(const FeatureVector& a, const FeatureVector& b);

	// 使用dlib库实现的匈牙利算法
	std::vector<std::pair<int, int>> hungarianAlgorithm(const std::vector<std::vector<double>>& cost_matrix);

	// 计算特征向量的局部密度（DPC算法的一部分）
	std::vector<double> calculateLocalDensity(const std::vector<FeatureVector>& features, double d_c);

	// 计算特征向量的相对距离（DPC算法的一部分）
	std::vector<double> calculateRelativeDistance(const std::vector<FeatureVector>& features, const std::vector<double>& rho);

	// 生成伪标签（格式：新目标-[ID]-近似类别-[元数据]）
	void generatePseudoLabels(std::vector<ClusterResult>& clusters, 
							  const std::map<std::shared_ptr<BaseObject>, std::vector<FeatureVector>>& historical_features,
		                      const std::vector<std::map<std::string, std::string>>& meta_datas);

	// 分析模型文件，返回模型信息（层数、参数量等）
	ModelInfo analyzeModel(const std::string& model_path);
};