#pragma once

#include "enums.h"
#include "includes.h"

// ���������ṹ�壬�������ִ�����������
struct FeatureVector {
    std::vector<double> visual_features;   // �Ӿ�����
    std::vector<double> radar_features;    // �״�����
    std::vector<double> audio_features;    // ��Ƶ����
    std::vector<double> shape_features;    // ��״����
    std::vector<double> motion_features;   // �˶�����

    // ������������Ƿ�Ϊ��
    bool isEmpty() const {
        return visual_features.empty() && radar_features.empty() &&
            audio_features.empty() && shape_features.empty() &&
            motion_features.empty();
    }
};

// �����������ļ����
struct Detection {
    std::string sensor_id;                  // ������ID
    SensorType sensor_type;                 // ����������
    Timestamp timestamp;                    // ʱ���
    int local_id;                           // ����������Ŀ��ID
    ObjectClass detected_class;             // ��⵽��Ŀ�����
    double class_confidence;                // ������Ŷ�
    Eigen::Vector3d position;               // ����������ϵ�µ�λ��
    Eigen::Vector3d position_global;        // ȫ������ϵ�µ�λ��
    Eigen::Vector3d velocity;               // ����������ϵ�µ��ٶ�
    Eigen::Vector3d velocity_global;        // ȫ������ϵ�µ��ٶ�
    cv::Rect2d bbox;                        // �߽��ͼ���ഫ������
    double detection_confidence;            // ������Ŷ�
    FeatureVector features;                 // Ŀ����������
    Eigen::Matrix3d covariance;             // λ�ò���Э�������
    Eigen::Matrix3d velocity_covariance;    // �ٶȲ���Э�������
};

// �ںϺ��Ŀ��
struct FusedObject {
    int global_id;                                      // ȫ��Ŀ��ID
    Timestamp timestamp;                                // ʱ���
    Eigen::Vector3d position;                           // ȫ��λ��
    Eigen::Vector3d velocity;                           // ȫ���ٶ�
    Eigen::Matrix3d position_covariance;                // λ��Э����
    Eigen::Matrix3d velocity_covariance;                // �ٶ�Э����
    std::map<ObjectClass, double> class_probabilities;  // �����ʷֲ�
    FeatureVector fused_features;                       // �ںϺ����������
    std::vector<Detection> associated_detections;       // �����ļ����
    int track_length;                                   // ���ٳ��ȣ�֡����
    bool is_new;                                        // �Ƿ�Ϊ��Ŀ��
	ObjectClass final_class;                            // �������
	double final_class_confidence;                      // ����������Ŷ�
    bool has_category_conflict;                         // �Ƿ��������ͻ
    //double conflict_score;                              // ��ͻ���֣�0~1��Խ�߳�ͻԽ���أ�
};

// ������У׼����
struct SensorCalibration {
	std::string sensor_id;                  // ������ID
	SensorType type;                        // ����������
    Eigen::Matrix3d rotation;               // ��ת����
    Eigen::Vector3d translation;            // ƽ������
    Eigen::Matrix3d covariance;             // covariance����
    std::chrono::microseconds time_offset;  // ʱ��ƫ��
    double time_drift;                      // ʱ��Ư��(ppm)����������ʱ�������������ʱ������ۻ���Ư�ƣ���ͨ���� ppm�������֮һ�� ��ʾ�����磬0.1ppm ��ʾÿ���� 1 �룬ʱ��������� 0.1 ΢�루1 �� �� 0.1/1e6��
};

// ����TensorRTִ�������ģ�ע������ָ���޷�������
struct Context {
	TRTUniquePtr<nvinfer1::IExecutionContext> context;	// TensorRTִ��������
	TRTUniquePtr<cudaStream_t>		stream;				// CUDA��
	TRTUniquePtr<cudaEvent_t>		event;				// CUDA�¼�
	void* hostOutput;					// �����������
	bool  isHostOutputUseCudaMemcpy;	// ������������Ƿ�ʹ��cudaMemcpy����
	void* deviceInput;					// �豸��������
	void* deviceOutput;					// �豸�������

	Context() : context(nullptr), stream(nullptr), event(nullptr), hostOutput(nullptr), deviceInput(nullptr), deviceOutput(nullptr), isHostOutputUseCudaMemcpy(false) {};

	// �ƶ����캯��
	Context(Context&& other) noexcept :
		context(std::move(other.context)),
		stream(std::move(other.stream)),
		event(std::move(other.event)),
		hostOutput(other.hostOutput),
		isHostOutputUseCudaMemcpy(false),
		deviceInput(other.deviceInput),
		deviceOutput(other.deviceOutput) {
		other.hostOutput = nullptr;
		other.deviceInput = nullptr;
		other.deviceOutput = nullptr;
	}

	// �ƶ���ֵ�����
	Context& operator=(Context&& other) noexcept {
		if (this != &other) {
			context = std::move(other.context);
			stream = std::move(other.stream);
			event = std::move(other.event);
			if (hostOutput != nullptr) {
				if(isHostOutputUseCudaMemcpy == true) {
					cudaFree(hostOutput);
				} else {
					delete[] hostOutput;
				}
			}
			if (deviceInput != nullptr) {
				cudaFree(deviceInput);
			}
			if (deviceOutput != nullptr) {
				cudaFree(deviceOutput);
			}
			hostOutput = other.hostOutput;
			isHostOutputUseCudaMemcpy = other.isHostOutputUseCudaMemcpy;
			deviceInput = other.deviceInput;
			deviceOutput = other.deviceOutput;
			other.hostOutput = nullptr;
			other.deviceInput = nullptr;
			other.deviceOutput = nullptr;
		}
		return *this;
	}

	~Context() {
		// ע�⣡��������cudaMemcpyϵ�к���������ڴ������cudaFree�ͷ�
		if (hostOutput != nullptr) {
			if (isHostOutputUseCudaMemcpy == true) {
				cudaFree(hostOutput);
			} else {
				delete[] hostOutput;
			}
			hostOutput = nullptr;
		}
		if (deviceInput != nullptr) {
			cudaFree(deviceInput);
			deviceInput = nullptr;
		}
		if (deviceOutput != nullptr) {
			cudaFree(deviceOutput);
			deviceOutput = nullptr;
		}
	}
};

// ͼ������
struct ImageDetectionResult {
	struct Box {
		cv::Rect rect;					// Ŀ�����
		int class_id;					// ���ID
		float confidence;				// ���Ŷ�
	};
	std::vector<Box> boxes;				// �洢�������Ŀ���
};

// α��ǩ�ṹ��
struct PseudoLabel {
    std::string label;															// α��ǩ�ַ������硰��Ŀ��-006-�������001-����-�̶���λ����
    std::shared_ptr<BaseObject> new_class;										// �����
	std::shared_ptr<BaseObject> associated_historical_class;					// ��������ʷ���
    std::map<std::string, std::string> metadata;								// �ؼ�Ԫ���ݣ��ٶȡ���λ�ȣ�
};

// �������ṹ��
struct ClusterResult {
	std::shared_ptr<BaseObject> cluster_class;						// �����
    std::vector<FeatureVector> samples;								// ������������
    FeatureVector cluster_center;									// ����������
    double avg_density;												// ����ƽ���ֲ��ܶ�
    PseudoLabel pseudo_label;										// ��Ӧ��α��ǩ
};

// �洢��̬��������ģ����Ϣ��ά�ȡ������������ͣ�
struct ModelInfo {
	int input_dim;                          // ģ������ά�ȣ���ģ̬������ά�ȣ�
	int output_dim;                         // ģ�����ά�ȣ��������
	int total_layer_num;                    // ģ���ܲ���
	struct LayerInfo {
		std::string layer_type;				// �����ͣ���Linear��ReLU�ȣ�
		int in_dim;							// ������ά��
		int out_dim;						// �����ά��
	};
	std::unordered_map<int, LayerInfo> layer_map;		// ÿ����Ϣ
};