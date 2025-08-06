#pragma once

#include "structs.h"

// TensorRT��־��
class TensorRTLogger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
		if (severity >= Severity::kWARNING) { // ֻ��ӡ���漰���ϼ���
			std::cerr << "TensorRT: " << msg << std::endl;
		}
	}
};

class TensorRT {
	std::vector<std::string> class_names_;			// ��������б�
	int input_width_;								// ������
	int input_height_;								// ����߶�
	float conf_threshold_;							// ���Ŷ���ֵ
	float nms_threshold_;							// �Ǽ���ֵ������ֵ
	size_t input_size_;								// ���뻺������С
	size_t output_size_;							// �����������С
	std::string input_tensor_name_;					// ������������
	std::string output_tensor_name_;				// �����������
	TRTUniquePtr<nvinfer1::IRuntime> runtime_;		// TensorRT����ʱ
	TRTUniquePtr<nvinfer1::ICudaEngine> engine_;	// TensorRT����
	std::vector<Context> contexts_;					// ִ���������б�
	TensorRTLogger gLogger;							// TensorRT��־��

public:
	TensorRT(const std::string& model_path, int inputW, int inputH, float conf_threshold, float nms_threshold);
	void setClassNames(const std::vector<std::string>& class_names) {
		class_names_ = class_names;
	}
	ImageDetectionResult detect(const cv::Mat& image);

private:
	virtual void createEngine(const std::string& model_path);
	virtual bool loadEngine(const std::string& engine_path);
	virtual void buildEngine(const std::string& model_path, const std::string& engine_path);
	virtual void saveEngine(const std::string& engine_path, nvinfer1::IHostMemory* serialized_engine);
	// Ԥ����
	virtual void preprocess(const cv::Mat& image, Context& context);
	// ����
	virtual void postprocess(const cv::Mat& image, const float* output, int output_size, ImageDetectionResult& detectionRes);
};