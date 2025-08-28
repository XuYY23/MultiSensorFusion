#pragma once

#include "structs.h"

class FeatureDataBase {
    using history_key = std::map<std::shared_ptr<BaseObject>, std::vector<FeatureVector>>;
	void init() {}
	INSTANCE(FeatureDataBase)
    PROPERTY(int, new_type_num, NewTypeNum)
    // ��ʷ���������������֪���ֵ�����������������
    PROPERTY(history_key, historical_features, HistoricalFeatures)
    // ���������Ŀ������
    PROPERTY(std::vector<FeatureVector>, new_features, NewFeatures)
    // �Ѿ���������
    PROPERTY(std::vector<ClusterResult>, new_categories, NewCategories)

public:
    // ����ʷ�����л�ȡÿ�����������ÿ��ѡk����
    std::vector<std::pair<std::shared_ptr<BaseObject>, FeatureVector>> getCoreHistoricalSamples(int k) const;

    // �����Ŀ��������������
    void addNewFeature(const FeatureVector& feat);

    // ������������ʷ����
    void updateHistoricalFeatures();

	// ʹ��DPC�㷨�����������о��࣬���ؾ�����
    std::vector<ClusterResult> performDPCClustering();
};