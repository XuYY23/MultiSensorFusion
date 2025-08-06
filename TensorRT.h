#pragma once

#include "structs.h"

// TensorRT日志器
class TensorRTLogger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
		if (severity >= Severity::kWARNING) { // 只打印警告及以上级别
			std::cerr << "TensorRT: " << msg << std::endl;
		}
	}
};

class TensorRT {
protected:
	bool is_engine_loaded_;							// 是否加载了引擎
	std::string model_path_;						// 模型路径
	std::vector<std::string> class_names_;			// 类别名称列表
	int input_width_;								// 输入宽度
	int input_height_;								// 输入高度
	float conf_threshold_;							// 置信度阈值
	float nms_threshold_;							// 非极大值抑制阈值
	size_t input_size_;								// 输入缓冲区大小
	size_t output_size_;							// 输出缓冲区大小
	std::string input_tensor_name_;					// 输入张量名称
	std::string output_tensor_name_;				// 输出张量名称
	TRTUniquePtr<nvinfer1::IRuntime> runtime_;		// TensorRT运行时
	TRTUniquePtr<nvinfer1::ICudaEngine> engine_;	// TensorRT引擎
	std::vector<Context> contexts_;					// 执行上下文列表
	TensorRTLogger gLogger;							// TensorRT日志器

public:
	TensorRT(const std::string& model_path, int inputW, int inputH, float conf_threshold, float nms_threshold);
	void setClassNames(const std::vector<std::string>& class_names) {
		class_names_ = class_names;
	}
	virtual ImageDetectionResult detect(const cv::Mat& image);
	virtual ~TensorRT() {
		contexts_.clear();
	}

protected:
	virtual void createEngine(const std::string& model_path);
	virtual bool loadEngine(const std::string& engine_path);
	virtual void buildEngine(const std::string& model_path, const std::string& engine_path);
	virtual void saveEngine(const std::string& engine_path, nvinfer1::IHostMemory* serialized_engine);
	// 预处理
	virtual void preprocess(const cv::Mat& image, Context& context);
	// 后处理
	virtual void postprocess(const cv::Mat& image, const float* output, int output_size, ImageDetectionResult& detectionRes);
};