#pragma once

#include "includes.h"

class Config {
	PROPERTY(double, position_weight_, PositionWeight)						// λ��Ȩ��
	PROPERTY(double, velocity_weight_, VelocityWeight)						// �ٶ�Ȩ��
	PROPERTY(double, class_weight_, ClassWeight)							// ���Ȩ��
	PROPERTY(double, feature_weight_, FeatureWeight)						// ����Ȩ��
	PROPERTY(double, max_position_distance_, MaxPositionDistance)			// ���λ�þ�����ֵ
	PROPERTY(double, max_velocity_diff_, MaxVelocityDiff)					// ����ٶȲ���ֵ
	PROPERTY(double, min_similarity_threshold_, MinSimilarityThreshold)		// ��С���ƶ���ֵ
	PROPERTY(double, conf_threshold, ConfThreshold)							// ���Ŷ���ֵ

private:
	Config() = default;							// ��ֹĬ�Ϲ��캯��
	Config(const Config&) = delete;				// ��ֹ��������
	Config& operator=(const Config&) = delete;	// ��ֹ������ֵ

public:
	~Config() {}
	static Config& GetInstance() {
		static Config config;
		return config;
	}
};