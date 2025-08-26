#pragma once

#include "structs.h"
#include "Config.h"
#include <torch/torch.h>

class IncrementalLearning {
	void init() {
		teacher_model_path      = Config::GetInstance().getTeacherModelPath();
        onnx_export_path        = Config::GetInstance().getOnnxExportPath();
        hist_core_sample_num    = Config::GetInstance().getHistCoreSampleNum();
        train_epochs            = Config::GetInstance().getTrainEpochs();
        learning_rate           = Config::GetInstance().getLearningRate();
        distill_T               = Config::GetInstance().getTemperature();
        distill_lambda1         = Config::GetInstance().getDistillLambda1();
        distill_lambda2         = Config::GetInstance().getDistillLambda2();
        distill_gamma           = Config::GetInstance().getIncrementalGamma();
        hist_acc_thresh         = Config::GetInstance().getHistoricalAccThreshold();
        new_acc_thresh          = Config::GetInstance().getNewClassAccThreshold();
    }
	INSTANCE(IncrementalLearning)

    torch::jit::script::Module teacher_model;       // 教师模型
    torch::jit::script::Module student_model;       // 学生模型
    ModelInfo teacher_model_info;                   // 教师模型的动态信息（维度、层数，解析得到）

    std::vector<torch::Tensor> train_feat_batch;    // 训练用特征（多模态拼接后的数据）
    std::vector<torch::Tensor> train_label_batch;   // 训练用标签（已知目标真实标签+新目标伪标签）
    std::vector<torch::Tensor> val_hist_feat;       // 验证用历史样本特征
    std::vector<torch::Tensor> val_hist_label;      // 验证用历史样本标签
    std::vector<torch::Tensor> val_new_feat;        // 验证用新目标样本特征
    std::vector<torch::Tensor> val_new_label;       // 验证用新目标样本标签

    std::string teacher_model_path;                 // 教师模型路径
    std::string onnx_export_path;                   // ONNX导出路径
    int hist_core_sample_num;                       // 每类历史样本取核心样本数
    int train_epochs;                               // 训练轮次
    float learning_rate;                            // 学习率
    float distill_T;                                // 蒸馏温度系数
    float distill_lambda1;                          // 输出层蒸馏权重（λ₁）
    float distill_lambda2;                          // 隐层蒸馏权重（λ₂）
    float distill_gamma;                            // 蒸馏损失总权重（γ）
    float hist_acc_thresh;                          // 历史目标准确率阈值
    float new_acc_thresh;                           // 新目标准确率阈值

public:
};