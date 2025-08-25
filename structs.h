#pragma once

#include "enums.h"
#include "includes.h"

// 特征向量结构体，包含各种传感器的特征
struct FeatureVector {
    std::vector<double> visual_features;   // 视觉特征
    std::vector<double> radar_features;    // 雷达特征
    std::vector<double> audio_features;    // 音频特征
    std::vector<double> shape_features;    // 形状特征
    std::vector<double> motion_features;   // 运动特征

    // 检查特征向量是否为空
    bool isEmpty() const {
        return visual_features.empty() && radar_features.empty() &&
            audio_features.empty() && shape_features.empty() &&
            motion_features.empty();
    }
};

// 单个传感器的检测结果
struct Detection {
    std::string sensor_id;                  // 传感器ID
    SensorType sensor_type;                 // 传感器类型
    Timestamp timestamp;                    // 时间戳
    int local_id;                           // 传感器本地目标ID
    ObjectClass detected_class;             // 检测到的目标类别
    double class_confidence;                // 类别置信度
    Eigen::Vector3d position;               // 传感器坐标系下的位置
    Eigen::Vector3d position_global;        // 全局坐标系下的位置
    Eigen::Vector3d velocity;               // 传感器坐标系下的速度
    Eigen::Vector3d velocity_global;        // 全局坐标系下的速度
    cv::Rect2d bbox;                        // 边界框（图像类传感器）
    double detection_confidence;            // 检测置信度
    FeatureVector features;                 // 目标特征向量
    Eigen::Matrix3d covariance;             // 位置测量协方差矩阵
    Eigen::Matrix3d velocity_covariance;    // 速度测量协方差矩阵
};

// 融合后的目标
struct FusedObject {
    int global_id;                                      // 全局目标ID
    Timestamp timestamp;                                // 时间戳
    Eigen::Vector3d position;                           // 全局位置
    Eigen::Vector3d velocity;                           // 全局速度
    Eigen::Matrix3d position_covariance;                // 位置协方差
    Eigen::Matrix3d velocity_covariance;                // 速度协方差
    std::map<ObjectClass, double> class_probabilities;  // 类别概率分布
    FeatureVector fused_features;                       // 融合后的特征向量
    std::vector<Detection> associated_detections;       // 关联的检测结果
    int track_length;                                   // 跟踪长度（帧数）
    bool is_new;                                        // 是否为新目标
	ObjectClass final_class;                            // 最终类别
	double final_class_confidence;                      // 最终类别置信度
    bool has_category_conflict;                         // 是否存在类别冲突
    //double conflict_score;                              // 冲突评分（0~1，越高冲突越严重）
};

// 传感器校准参数
struct SensorCalibration {
	std::string sensor_id;                  // 传感器ID
	SensorType type;                        // 传感器类型
    Eigen::Matrix3d rotation;               // 旋转矩阵
    Eigen::Vector3d translation;            // 平移向量
    Eigen::Matrix3d covariance;             // covariance矩阵
    std::chrono::microseconds time_offset;  // 时间偏移
    double time_drift;                      // 时间漂移(ppm)，传感器的时间测量会随运行时间产生累积误差（漂移），通常用 ppm（百万分之一） 表示。例如，0.1ppm 表示每运行 1 秒，时间误差增加 0.1 微秒（1 秒 × 0.1/1e6）
};

// 单个TensorRT执行上下文，注意智能指针无法被拷贝
struct Context {
	TRTUniquePtr<nvinfer1::IExecutionContext> context;	// TensorRT执行上下文
	TRTUniquePtr<cudaStream_t>		stream;				// CUDA流
	TRTUniquePtr<cudaEvent_t>		event;				// CUDA事件
	void* hostOutput;					// 主机输出数据
	bool  isHostOutputUseCudaMemcpy;	// 主机输出数据是否使用cudaMemcpy分配
	void* deviceInput;					// 设备输入数据
	void* deviceOutput;					// 设备输出数据

	Context() : context(nullptr), stream(nullptr), event(nullptr), hostOutput(nullptr), deviceInput(nullptr), deviceOutput(nullptr), isHostOutputUseCudaMemcpy(false) {};

	// 移动构造函数
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

	// 移动赋值运算符
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
		// 注意！！！，用cudaMemcpy系列函数分配的内存必须用cudaFree释放
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

// 图像检测结果
struct ImageDetectionResult {
	struct Box {
		cv::Rect rect;					// 目标检测框
		int class_id;					// 类别ID
		float confidence;				// 置信度
	};
	std::vector<Box> boxes;				// 存储检测结果的目标框
};

// 伪标签结构体
struct PseudoLabel {
    std::string label;															// 伪标签字符串（如“新目标-006-近似类别001-低速-固定方位”）
    std::shared_ptr<BaseObject> new_class;										// 新类别
	std::shared_ptr<BaseObject> associated_historical_class;					// 关联的历史类别
    std::map<std::string, std::string> metadata;								// 关键元数据（速度、方位等）
};

// 聚类结果结构体
struct ClusterResult {
	std::shared_ptr<BaseObject> cluster_class;						// 簇类别
    std::vector<FeatureVector> samples;								// 簇内样本特征
    FeatureVector cluster_center;									// 簇中心特征
    double avg_density;												// 簇内平均局部密度
    PseudoLabel pseudo_label;										// 对应的伪标签
};

// 增量模型配置结构体
struct IncrementalModelConfig {
    std::string teacher_model_path;     // 教师模型路径
    std::string student_model_path;     // 学生模型路径
    float temperature = 5.0f;           // 知识蒸馏温度系数
    float lambda1 = 0.7f;               // 输出层蒸馏损失权重
    float lambda2 = 0.3f;               // 隐层蒸馏损失权重
    float gamma = 0.5f;                 // 蒸馏损失在总损失中的权重
};