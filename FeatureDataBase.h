#pragma once

#include "structs.h"

class FeatureDataBase {
    using history_key = std::map<std::shared_ptr<BaseObject>, std::vector<FeatureVector>>;
	void init() {}
	INSTANCE(FeatureDataBase)
    PROPERTY(int, new_type_num, NewTypeNum)
    // 历史类别特征（键：已知类别，值：该类别所有特征）
    PROPERTY(history_key, historical_features, HistoricalFeatures)
    // 待聚类的新目标特征
    PROPERTY(std::vector<FeatureVector>, new_features, NewFeatures)
    // 已聚类的新类别
    PROPERTY(std::vector<ClusterResult>, new_categories, NewCategories)

public:
    // 从历史特征中获取每类核心样本（每类选k个）
    std::vector<std::pair<std::shared_ptr<BaseObject>, FeatureVector>> getCoreHistoricalSamples(int k) const;

    // 添加新目标特征到特征库
    void addNewFeature(const FeatureVector& feat);

    // 将新类别加入历史特征
    void updateHistoricalFeatures();

	// 使用DPC算法对新特征进行聚类，返回聚类结果
    std::vector<ClusterResult> performDPCClustering();
};