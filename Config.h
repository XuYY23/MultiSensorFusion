#pragma once

#include "includes.h"

class Config {
	void init() {}															// ��ʼ������
	INSTANCE(Config)
	PROPERTY(double, position_weight_, PositionWeight)						// λ��Ȩ��
	PROPERTY(double, velocity_weight_, VelocityWeight)						// �ٶ�Ȩ��
	PROPERTY(double, class_weight_, ClassWeight)							// ���Ȩ��
	PROPERTY(double, feature_weight_, FeatureWeight)						// ����Ȩ��
	PROPERTY(double, max_position_distance_, MaxPositionDistance)			// ���λ�þ�����ֵ
	PROPERTY(double, max_velocity_diff_, MaxVelocityDiff)					// ����ٶȲ���ֵ
	PROPERTY(double, min_similarity_threshold_, MinSimilarityThreshold)		// ��С���ƶ���ֵ
	PROPERTY(double, conf_threshold, ConfThreshold)							// ���Ŷ���ֵ
	PROPERTY(double, kl_div_threshold_, KlDivThreshold)						// KLɢ����ֵ
	PROPERTY(double, new_target_cosine_thresh_, NewTargetCosineThresh)		// ��Ŀ���������ƶ���ֵ
	PROPERTY(int, new_sample_cluster_threshold_, NewSampleClusterThreshold)	// ���ഥ��������
	PROPERTY(double, dpc_rho_min_, DpcRhoMin)								// DPC�ֲ��ܶ���Сֵ
	PROPERTY(double, dpc_delta_min_, DpcDeltaMin)							// DPC��Ծ�����Сֵ
	PROPERTY(double, d_c, CutDistance)										// DPC�ضϾ���
	PROPERTY(double, isolated_point_min, IsolatedPointMin)					// ��������ֵ
	PROPERTY(double, incremental_gamma_, IncrementalGamma)					// ����ѵ������ʧȨ�ئ�
	PROPERTY(double, distill_lambda1_, DistillLambda1)						// ������ʧ��1������㣩
	PROPERTY(double, distill_lambda2_, DistillLambda2)						// ������ʧ��2�����㣩
	PROPERTY(double, temperature, Temperature)								// �����¶�ϵ��
	PROPERTY(double, historical_acc_threshold_, HistoricalAccThreshold)		// ��ʷĿ��׼ȷ����ֵ
	PROPERTY(double, new_class_acc_threshold_, NewClassAccThreshold)		// ��Ŀ��׼ȷ����ֵ
	PROPERTY(int, inc_learning_time_gap, TimeGap)							// ����ѧϰ�߳�ÿ��˯��ʱ�����룩
	PROPERTY(int, inc_learning_num_gap, NumGap)								// ����ѧϰ��������������
	PROPERTY(std::string, teacher_model_path, TeacherModelPath)				// ��ʦģ��·��
	PROPERTY(std::string, student_model_path, StudentModelPath)				// ѧ��ģ��·��
	PROPERTY(int, default_model_input_dim, DefaultInputDim)					// Ĭ��ģ������ά��
	PROPERTY(int, default_model_output_dim, DefaultOutputDim)				// Ĭ��ģ�����ά��
	PROPERTY(std::string, linear_type, LinearType)							// ȫ���Ӳ����������torch.nn.modules.linear.Linear��
	PROPERTY(int, hist_core_sample_num, HistCoreSampleNum)					// ÿ����ʷ����ȡ����������
	PROPERTY(int, train_epochs, TrainEpochs)								// ѵ���ִ�
	PROPERTY(double, learning_rate, LearningRate)							// ѧϰ��
	PROPERTY(std::string, hidden_feat, HiddenFeat)							// ����������
	PROPERTY(std::string, onnx_import_path, OnnxImportPath)					// onnx����·��
	PROPERTY(std::string, onnx_export_path, OnnxExportPath)					// onnx����·��
	PROPERTY(std::string, onnx_input_name, OnnxInputName)					// ����onnxģ�͵�������
	PROPERTY(std::string, onnx_output_name, OnnxOutputName)					// ����onnxģ�͵������
	PROPERTY(std::string, export_model_py_file_path, ExportModelPyFilePath)	// ����onnxģ�͵�python�ű�·��
};