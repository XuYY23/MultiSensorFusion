#include "FeatureDataBase.h"
#include "Config.h"
#include "MathUtils.h"
#include "ReuseFunction.h"

// ����ʷ�����л�ȡÿ�����������ÿ��ѡk����
std::vector<std::pair<std::shared_ptr<BaseObject>, FeatureVector>> FeatureDataBase::getCoreHistoricalSamples(int k) const {
    std::vector<std::pair<std::shared_ptr<BaseObject>, FeatureVector>> core_samples;
    for (const auto& [cls, feats] : historical_features) {
        if (feats.empty()) {
            continue;
        }
        int take = std::min(k, (int)feats.size());
        for (int i = 0; i < take; ++i) {
            core_samples.emplace_back(cls, feats[i]);
        }
    }
    return core_samples;
}

// �����Ŀ��������������
void FeatureDataBase::addNewFeature(const FeatureVector& feat) {
    new_features.push_back(feat);
}

// ������������ʷ����
void FeatureDataBase::updateHistoricalFeatures() {
    for (const auto& cluster : new_categories) {
        for (const FeatureVector& sample : cluster.samples) {
            historical_features[cluster.cluster_class].push_back(sample);
        }
    }
    new_features.clear();
    new_categories.clear();
}

std::vector<ClusterResult> FeatureDataBase::performDPCClustering() {
	double d_c = Config::GetInstance().getCutDistance();
    // ����ֲ��ܶȺ���Ծ���
	std::vector<double> rho = ReuseFunction::GetInstance().calculateLocalDensity(new_features, d_c);
	std::vector<double> delta = ReuseFunction::GetInstance().calculateRelativeDistance(new_features, rho);
    // ��ȡ������
	const double rho_min = Config::GetInstance().getDpcRhoMin();
	const double delta_min = Config::GetInstance().getDpcDeltaMin();
    size_t size = new_features.size();
    std::vector<int> cluster_center_sample_num;
    std::vector<ClusterResult> cluster_center;
    for (size_t i = 0; i < size; ++i) {
        if (rho[i] > rho_min && delta[i] > delta_min) {
            ClusterResult center;
            center.cluster_class = std::make_shared<PotentialNewTypeObject>();
            center.samples.push_back(new_features[i]);
            center.cluster_center = new_features[i];
            center.avg_density = rho[i];
            cluster_center.push_back(center);
            cluster_center_sample_num.push_back(1);
        }
    }

    // ����������������
    const double isolated_point_min = Config::GetInstance().getIsolatedPointMin();
    for (size_t i = 0; i < size; ++i) {
		double min_dis = std::numeric_limits<double>::max();
        int center_idx = -1;
        for (size_t j = 0; j < cluster_center.size(); ++j) {
            double dis = 1 - ReuseFunction::GetInstance().calculateFeatureSimilarity(new_features[i], cluster_center[j].cluster_center);
            if (dis <= isolated_point_min && dis < min_dis) {
                min_dis = dis;
                center_idx = j;
            }
        }

        if (center_idx != -1) {
            cluster_center[center_idx].avg_density += rho[i];
            cluster_center[center_idx].samples.push_back(new_features[i]);
            ++cluster_center_sample_num[center_idx];
        } else {    // ������
            ClusterResult center;
            center.cluster_class = std::make_shared<BaseObject>();
            center.samples.push_back(new_features[i]);
            center.cluster_center = new_features[i];
            center.avg_density = rho[i];
            cluster_center.push_back(center);
            cluster_center_sample_num.push_back(1);
        }
    }

    for (ClusterResult& cluster_result : cluster_center) {
        cluster_result.avg_density /= cluster_result.samples.size();
    }

    return cluster_center;
}
