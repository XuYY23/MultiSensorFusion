#include "ReuseFunction.h"
#include "MathUtils.h"
#include "Config.h"
#include <torch/script.h>
#include <fstream>

// ������������������ƶ�
double ReuseFunction::calculateSimilarity(const Detection& a, const Detection& b) {
    // 1. λ�����ƶ� (ʹ�ø�˹��)
    double pos_dist = (a.position_global - b.position_global).norm();
    double pos_sim = MathUtils::GetInstance().gaussianSimilarity(pos_dist, Config::GetInstance().getMaxPositionDistance());

    // 2. �ٶ����ƶ�
    double vel_diff = (a.velocity_global - b.velocity_global).norm();
    double vel_sim = MathUtils::GetInstance().gaussianSimilarity(vel_diff, Config::GetInstance().getMaxVelocityDiff());

    // 3. ������ƶ�
    double class_sim = (a.detected_class == b.detected_class) ? 1.0 : 0.0;
    // ����������Ŷ�
    class_sim *= (a.class_confidence + b.class_confidence) * 0.5;

    // 4. �������ƶ� (ʹ���������ƶ�)
    double feat_sim = calculateFeatureSimilarity(a.features, b.features);

    // ��Ȩ�ں����ƶ�
    double similarity = Config::GetInstance().getPositionWeight() * pos_sim
        + Config::GetInstance().getVelocityWeight() * vel_sim
        + Config::GetInstance().getClassWeight() * class_sim
        + Config::GetInstance().getFeatureWeight() * feat_sim;

    return similarity;
}

// �����������������ƶ�
double ReuseFunction::calculateFeatureSimilarity(const FeatureVector& a, const FeatureVector& b) {
    // �����������������Ϊ�գ����ƶ�Ϊ1.0
    if (a.isEmpty() && b.isEmpty()) {
        return 1.0;
    }

    // ���һ��Ϊ�գ�һ����Ϊ�գ����ƶ�Ϊ0.0
    if (a.isEmpty() || b.isEmpty()) {
        return 0.0;
    }

    double total_sim = 0.0;
    int feature_count = 0;

    // �Ӿ��������ƶ�
    if (!a.visual_features.empty() && !b.visual_features.empty()) {
        total_sim += MathUtils::GetInstance().cosineSimilarity(a.visual_features, b.visual_features);
        feature_count++;
    }

    // �״��������ƶ�
    if (!a.radar_features.empty() && !b.radar_features.empty()) {
        total_sim += MathUtils::GetInstance().cosineSimilarity(a.radar_features, b.radar_features);
        feature_count++;
    }

    // ��Ƶ�������ƶ�
    if (!a.audio_features.empty() && !b.audio_features.empty()) {
        total_sim += MathUtils::GetInstance().cosineSimilarity(a.audio_features, b.audio_features);
        feature_count++;
    }

    // ��״�������ƶ�
    if (!a.shape_features.empty() && !b.shape_features.empty()) {
        total_sim += MathUtils::GetInstance().cosineSimilarity(a.shape_features, b.shape_features);
        feature_count++;
    }

    // �˶��������ƶ�
    if (!a.motion_features.empty() && !b.motion_features.empty()) {
        total_sim += MathUtils::GetInstance().cosineSimilarity(a.motion_features, b.motion_features);
        feature_count++;
    }

    // ����ƽ�����ƶ�
    return feature_count > 0 ? total_sim / feature_count : 0.0;
}

// ʹ��dlib��ʵ���������㷨��20.0�汾��
std::vector<std::pair<int, int>> ReuseFunction::hungarianAlgorithm(const std::vector<std::vector<double>>& cost_matrix) {
    std::vector<std::pair<int, int>> result;

    if (cost_matrix.empty() || cost_matrix[0].empty()) {
        return result;
    }

    // ��ȡ�����С
    int rows = cost_matrix.size();
    int cols = cost_matrix[0].size();

    // dlib�����ɱ������㷨��Ҫ����������Ƿ��������
    int n = std::max(rows, cols);

    // �����ɱ�����dlibʹ��������⣬����������һ����ֵ��ȥ�ɱ���
    dlib::matrix<int> assignment_matrix(n, n);
    const int MAX_COST = 1000000;  // �㹻��ĳ���

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i < rows && j < cols) {
                // ת��Ϊ��������ת�ɱ�����Ϊdlibʵ�ֵ������ɱ����䣩
                assignment_matrix(i, j) = static_cast<int>(MAX_COST - cost_matrix[i][j] * MAX_COST);
            }
            else {
                // ����λ�óɱ���Ϊ0
                assignment_matrix(i, j) = 0;
            }
        }
    }

    // ִ�����ɱ������㷨
    std::vector<long> assignment = max_cost_assignment(assignment_matrix);

    // ��������ֻ������Чƥ��
    for (int i = 0; i < rows; ++i) {
        if (assignment[i] < cols) {  // ֻ��������Ч��Χ�ڵ�ƥ��
            // ���ƥ���Ƿ���Ч���ɱ�������ֵ��
            if (cost_matrix[i][assignment[i]] < (1.0 - Config::GetInstance().getMinSimilarityThreshold())) {
                result.emplace_back(i, assignment[i]);
            }
        }
    }

    return result;
}

std::vector<double> ReuseFunction::calculateLocalDensity(const std::vector<FeatureVector>& features, double d_c) {
    std::vector<double> rho(features.size(), 0.0);
    for (size_t i = 0; i < features.size(); ++i) {
        for (size_t j = 0; j < features.size(); ++j) {
            if (i == j) {
                continue;
            }

            // ����=1-�������ƶ�
            double dist = 1 - calculateFeatureSimilarity(features[i], features[j]);
            if (dist < d_c) {
                rho[i] += 1.0; // ָʾ������(d_ij - d_c)
            }
        }
    }
    return rho;
}

std::vector<double> ReuseFunction::calculateRelativeDistance(const std::vector<FeatureVector>& features, const std::vector<double>& rho) {
    std::vector<double> delta(features.size(), 0.0);
    double max_rho = *std::max_element(rho.begin(), rho.end());

    for (size_t i = 0; i < features.size(); ++i) {
        if (rho[i] == max_rho) {
            // ȫ���ܶ���󣬦�ȡ������
            double max_dist = 0.0;
            for (size_t j = 0; j < features.size(); ++j) {
                if (i == j) {
                    continue;
                }
                double dist = 1 - calculateFeatureSimilarity(features[i], features[j]);
                max_dist = std::max(max_dist, dist);
            }
            delta[i] = max_dist;
        } else {
            // �Ҧ�>��ǰ��������С����
            double min_dist = 1e9;
            for (size_t j = 0; j < features.size(); ++j) {
                if (rho[j] <= rho[i]) {
                    continue;
                }
                double dist = 1 - calculateFeatureSimilarity(features[i], features[j]);
                min_dist = std::min(min_dist, dist);
            }
            delta[i] = min_dist;
        }
    }
    return delta;
}

void ReuseFunction::generatePseudoLabels(std::vector<ClusterResult>& clusters,
                                         const std::map<std::shared_ptr<BaseObject>, std::vector<FeatureVector>>& historical_features,
                                         const std::vector<std::map<std::string, std::string>>& meta_datas) {
    int idx = 0;
    for (ClusterResult& cluster : clusters) {
        PseudoLabel label;
        label.new_class = cluster.cluster_class;

        // �ҹ�������ʷ���Top1���ƶȣ�
        double max_sim = 0.0;
        std::shared_ptr<BaseObject> associated_cls = std::make_shared<BaseObject>();
        for (const auto& [cls, hist_feats] : historical_features) {
            if (hist_feats.empty()) {
                continue;
            }
            // �������������ʷ������ƽ�����ƶ�
            double avg_sim = 0.0;
            for (const auto& feat : hist_feats) {
                avg_sim += calculateFeatureSimilarity(cluster.cluster_center, feat);
            }
            avg_sim /= hist_feats.size();
            if (avg_sim > max_sim) {
                max_sim = avg_sim;
                associated_cls = cls;
            }
        }
        label.associated_historical_class = associated_cls;
        if(idx >= meta_datas.size()) {
            std::cerr << "Warning: Metadata size is less than clusters size." << std::endl;
            label.metadata = {};
        } else {
            label.metadata = meta_datas[idx];
        }
		
        // ��װα��ǩ�ַ���
        label.label = "��Ŀ��-" + label.new_class->toString() + "-�������" + label.associated_historical_class->toString();
        for (const auto& [key, value] : label.metadata) {
			label.label += "-" + value;
        }
        cluster.pseudo_label = label;
        ++idx;
    }
}

ModelInfo ReuseFunction::analyzeModel(const std::string& model_path) {
    ModelInfo model_info;

    try {
        // ����JITģ�ͣ�������ǰ֪���ṹ��ֱ�Ӷ�ģ���ļ���
        torch::jit::script::Module model = torch::jit::load(model_path);
        model.eval(); // ��Ϊ����ģʽ������Ӱ��ģ����Ϣ����

        // ����ģ��������ģ��
        int layer_idx = 0;
        for (const auto& [layer_name, sub_module] : model.named_modules()) {
            // ������ģ�飨����ͳ����Ч�㣩
            if (sub_module.parameters().size() == 0) {
                continue;
            }

            // ʶ������ͣ��˴��ԡ�ȫ���Ӳ�Linear��Ϊ������ģ̬ģ�ͺ��Ĳ㣩�жϵ�ǰ���ǡ�ȫ���Ӳ㡱���������㣨��ReLU����㣩
            std::string layer_type = sub_module.type()->name().value().name();
            model_info.layer_map[layer_idx].layer_type = layer_type; // ��¼��N�������
            model_info.total_layer_num++; // �ܲ���+1

            // �Ƶ�����������ά��
            // ȫ���Ӳ��Ȩ��shape�ǣ�[���ά��, ����ά��]����Ȩ��shapeΪ[512,2048]����in=2048, out=512��
            if (layer_type == Config::GetInstance().getLinearType()) {
                // ��ȡȫ���Ӳ��Ȩ�ز���
                torch::Tensor weight = sub_module.attr("weight").toTensor();
                int layer_in_dim = weight.size(1);  // Ȩ�ص�2ά=����ά��
                int layer_out_dim = weight.size(0); // Ȩ�ص�1ά=���ά��
                model_info.layer_map[layer_idx].in_dim = layer_in_dim;
                model_info.layer_map[layer_idx].out_dim = layer_out_dim;

                // ��¼ģ����������ά�ȣ�ȡ��һ��ȫ���Ӳ������ά�ȣ�
                if (layer_idx == 0) {
                    model_info.input_dim = layer_in_dim;
                }
                // ��¼ģ���������ά�ȣ�ȡ���һ��ȫ���Ӳ�����ά�ȣ�
                model_info.output_dim = layer_out_dim;
            } else if (layer_type == "") {  // ����������

            }

            layer_idx++;
        }

        // ��ӡ�������
        std::cout << "=== ģ�ͽ������ ===" << std::endl;
		std::cout << "ģ��·����" << model_path << std::endl;
        std::cout << "ģ���ܲ�����" << model_info.total_layer_num << std::endl;
        std::cout << "ģ������ά�ȣ���ģ̬������ά�ȣ���" << model_info.input_dim << std::endl;
        std::cout << "ģ�����ά�ȣ����������" << model_info.output_dim << std::endl;
        for (int i = 0; i < model_info.total_layer_num; i++) {
            std::cout << "��" 
                      << i + 1 
                      << "�㣺����=" 
                      << model_info.layer_map[i].layer_type
                      << "������ά��=" << model_info.layer_map[i].in_dim
                      << "�����ά��=" << model_info.layer_map[i].out_dim 
                      << std::endl;
        }

    } catch (const c10::Error& e) {
        throw e;
    }

    return model_info;
}

torch::Tensor ReuseFunction::convertFeatureToTensor(const FeatureVector& feat, int model_input_dim) {
    if (model_input_dim <= 0) {
        model_input_dim = Config::GetInstance().getDefaultInputDim();
    }
    // �洢����ģ̬������
    std::vector<float> all_feats;

    // ƴ���Ӿ�����
    all_feats.insert(all_feats.end(), feat.visual_features.begin(), feat.visual_features.end());
    // ƴ���״�����
    all_feats.insert(all_feats.end(), feat.radar_features.begin(), feat.radar_features.end());
    // ƴ����Ƶ����
    all_feats.insert(all_feats.end(), feat.audio_features.begin(), feat.audio_features.end());
	// ƴ����״����
    all_feats.insert(all_feats.end(), feat.shape_features.begin(), feat.shape_features.end());
    // ƴ���˶�����
    all_feats.insert(all_feats.end(), feat.motion_features.begin(), feat.motion_features.end());

    // ���㵽��ά�ȣ�ȷ��ÿ������������ά��һ�£�ģ�Ͳ��ܴ���
    while (all_feats.size() < model_input_dim) {
        all_feats.push_back(0.0f);
    }

    // �ضϳ���ά�ȵĲ��֣���������������
    if (all_feats.size() > model_input_dim) {
        all_feats.resize(model_input_dim);
    }

    // ת��ΪTensor��ģ���ܴ���ĸ�ʽ����1��������total_dimά
    return torch::from_blob(all_feats.data(), { 1, model_input_dim }, torch::kFloat32);
}

bool ReuseFunction::loadJsonConfig(const std::string& file_path, json& config) {
    try {
        std::ifstream ifs(file_path);
        if (!ifs.is_open()) {
            std::cerr << "�޷���JSON�ļ�: " << file_path << std::endl;
            return false;
        }

        ifs >> config;  // ���ļ�����ȡ������JSON
        ifs.close();

        std::cout << "JSON�ļ���ȡ�ɹ�: " << file_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "JSON��������: " << e.what() << std::endl;
        return false;
    }
}
