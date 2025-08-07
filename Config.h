#pragma once

#include "includes.h"

class Config {
	PROPERTY(double, position_weight_, PositionWeight)						// 位置权重
	PROPERTY(double, velocity_weight_, VelocityWeight)						// 速度权重
	PROPERTY(double, class_weight_, ClassWeight)							// 类别权重
	PROPERTY(double, feature_weight_, FeatureWeight)						// 特征权重
	PROPERTY(double, max_position_distance_, MaxPositionDistance)			// 最大位置距离阈值
	PROPERTY(double, max_velocity_diff_, MaxVelocityDiff)					// 最大速度差阈值
	PROPERTY(double, min_similarity_threshold_, MinSimilarityThreshold)		// 最小相似度阈值
	PROPERTY(double, conf_threshold, ConfThreshold)							// 置信度阈值

private:
	Config() = default;							// 禁止默认构造函数
	Config(const Config&) = delete;				// 禁止拷贝构造
	Config& operator=(const Config&) = delete;	// 禁止拷贝赋值

public:
	~Config() {}
	static Config& GetInstance() {
		static Config config;
		return config;
	}
};