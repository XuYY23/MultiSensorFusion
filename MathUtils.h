#pragma once

#include "structs.h"

class MathUtils {
	void init() {}
	INSTANCE(MathUtils)
public:
	// ���������������������ƶ�
	double cosineSimilarity(const Eigen::Vector3d& a, const Eigen::Vector3d& b);
	double cosineSimilarity(const std::vector<double>& a, const std::vector<double>& b);

	// �����˹���ƶȣ�����λ�ú��ٶȵ����Ͼ��룩
	double gaussianSimilarity(const double& dis, const double& scale);

	// ����KLɢ�ȣ������Ĳ���̶ȣ���ֵԽС˵����������Խ���ƣ�
	double calculateKL(const FeatureVector& a, const FeatureVector& b);
};