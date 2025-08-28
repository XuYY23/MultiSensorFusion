#pragma once

#include "includes.h"

class Config {
	void init() {}															// 初始化函数
	INSTANCE(Config)
	PROPERTY(double, position_weight_, PositionWeight)						// 位置权重
	PROPERTY(double, velocity_weight_, VelocityWeight)						// 速度权重
	PROPERTY(double, class_weight_, ClassWeight)							// 类别权重
	PROPERTY(double, feature_weight_, FeatureWeight)						// 特征权重
	PROPERTY(double, max_position_distance_, MaxPositionDistance)			// 最大位置距离阈值
	PROPERTY(double, max_velocity_diff_, MaxVelocityDiff)					// 最大速度差阈值
	PROPERTY(double, min_similarity_threshold_, MinSimilarityThreshold)		// 最小相似度阈值
	PROPERTY(double, conf_threshold, ConfThreshold)							// 置信度阈值
	PROPERTY(double, kl_div_threshold_, KlDivThreshold)						// KL散度阈值
	PROPERTY(double, new_target_cosine_thresh_, NewTargetCosineThresh)		// 新目标余弦相似度阈值
	PROPERTY(int, new_sample_cluster_threshold_, NewSampleClusterThreshold)	// 聚类触发样本数
	PROPERTY(double, dpc_rho_min_, DpcRhoMin)								// DPC局部密度最小值
	PROPERTY(double, dpc_delta_min_, DpcDeltaMin)							// DPC相对距离最小值
	PROPERTY(double, d_c, CutDistance)										// DPC截断距离
	PROPERTY(double, isolated_point_min, IsolatedPointMin)					// 孤立点阈值
	PROPERTY(double, incremental_gamma_, IncrementalGamma)					// 增量训练总损失权重γ
	PROPERTY(double, distill_lambda1_, DistillLambda1)						// 蒸馏损失λ1（输出层）
	PROPERTY(double, distill_lambda2_, DistillLambda2)						// 蒸馏损失λ2（隐层）
	PROPERTY(double, temperature, Temperature)								// 蒸馏温度系数
	PROPERTY(double, historical_acc_threshold_, HistoricalAccThreshold)		// 历史目标准确率阈值
	PROPERTY(double, new_class_acc_threshold_, NewClassAccThreshold)		// 新目标准确率阈值
	PROPERTY(int, inc_learning_time_gap, TimeGap)							// 增量学习线程每次睡眠时长（秒）
	PROPERTY(int, inc_learning_num_gap, NumGap)								// 增量学习函数触发样本数
	PROPERTY(std::string, teacher_model_path, TeacherModelPath)				// 教师模型路径
	PROPERTY(std::string, student_model_path, StudentModelPath)				// 学生模型路径
	PROPERTY(int, default_model_input_dim, DefaultInputDim)					// 默认模型输入维度
	PROPERTY(int, default_model_output_dim, DefaultOutputDim)				// 默认模型输出维度
	PROPERTY(std::string, linear_type, LinearType)							// 全连接层的类型名（torch.nn.modules.linear.Linear）
	PROPERTY(int, hist_core_sample_num, HistCoreSampleNum)					// 每类历史样本取核心样本数
	PROPERTY(int, train_epochs, TrainEpochs)								// 训练轮次
	PROPERTY(double, learning_rate, LearningRate)							// 学习率
	PROPERTY(std::string, hidden_feat, HiddenFeat)							// 隐层特征名
	PROPERTY(std::string, onnx_import_path, OnnxImportPath)					// onnx导入路径
	PROPERTY(std::string, onnx_export_path, OnnxExportPath)					// onnx导出路径
	PROPERTY(std::string, onnx_input_name, OnnxInputName)					// 导出onnx模型的输入名
	PROPERTY(std::string, onnx_output_name, OnnxOutputName)					// 导出onnx模型的输出名
	PROPERTY(std::string, export_model_py_file_path, ExportModelPyFilePath)	// 导出onnx模型的python脚本路径
};