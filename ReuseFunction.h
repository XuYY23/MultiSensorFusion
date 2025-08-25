#pragma once

#include "structs.h"

class ReuseFunction {
	void init() {}
	INSTANCE(ReuseFunction)

public:
	// ������������������ƶ�
	double calculateSimilarity(const Detection& a, const Detection& b);

	// �����������������ƶ�
	double calculateFeatureSimilarity(const FeatureVector& a, const FeatureVector& b);

	// ʹ��dlib��ʵ�ֵ��������㷨
	std::vector<std::pair<int, int>> hungarianAlgorithm(const std::vector<std::vector<double>>& cost_matrix);

	// �������������ľֲ��ܶȣ�DPC�㷨��һ���֣�
	std::vector<double> calculateLocalDensity(const std::vector<FeatureVector>& features, double d_c);

	// ����������������Ծ��루DPC�㷨��һ���֣�
	std::vector<double> calculateRelativeDistance(const std::vector<FeatureVector>& features, const std::vector<double>& rho);

	// ����α��ǩ����ʽ����Ŀ��-[ID]-�������-[Ԫ����]��
	void generatePseudoLabels(std::vector<ClusterResult>& clusters, 
							  const std::map<std::shared_ptr<BaseObject>, std::vector<FeatureVector>>& historical_features,
		                      const std::vector<std::map<std::string, std::string>>& meta_datas);
};