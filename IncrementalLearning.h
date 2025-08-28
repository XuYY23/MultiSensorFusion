#pragma once

#include "structs.h"
#include "Config.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/serialization/export.h>

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
    std::vector<torch::Tensor> trainable_params;    // 存储可训练参数（仅分类头）

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
    // 增量训练入口
    void startIncrementalTraining();

private:
    // 准备训练/验证样本
    // 功能：从特征库拿“历史核心样本”和“新聚类样本”，整理成模型能读的格式
    bool prepareTrainValSamples();

    // 加载教师模型并动态解析结构
    // 功能：加载已保存的教师模型，同时解析出它的维度、层数
    bool loadAndParseTeacherModel();

    // 初始化学生模型
    // 功能：复制教师模型结构，只改最后一层（分类头）适配新类别数
    bool initStudentModel(int new_class_num);

    // 执行增量训练
    // 功能：让学生模型学新目标，同时不忘记教师模型的历史知识
    void runTrainingLoop();

    // 模型验证
    // 功能：检查学生模型是否“没忘旧知识”（历史样本准）且“学会新知识”（新样本准）
    bool validateStudentModel();

    // 保存模型
    void saveModel();

    // 导出为onnx格式
    void exportModelToONNX();

    // 统计新类别数量（从特征库的新聚类结果获取）
    int countNewClassNum();

    // 伪标签转整数ID（新目标伪标签→模型能算的数字标签）
    int convertPseudoLabelToId(const PseudoLabel& pseudo_label);
};