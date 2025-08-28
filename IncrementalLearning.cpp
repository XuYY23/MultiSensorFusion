#include "IncrementalLearning.h"
#include "FeatureDataBase.h"
#include "ReuseFunction.h"

void IncrementalLearning::startIncrementalTraining() {
    std::cout << "=== 开始增量学习流程 ===" << std::endl;

    // 加载教师模型并解析结构
    if (!loadAndParseTeacherModel()) {
        return;
    }

    // 统计新类别数量，初始化学生模型
    int new_class_num = countNewClassNum();
    if (!initStudentModel(new_class_num)) {
        return;
    }

    // 准备训练/验证样本
    if (!prepareTrainValSamples()) {
        return;
    }

    // 执行训练循环
    runTrainingLoop();

    // 验证学生模型
    if (!validateStudentModel()) {
        return;
    }

    // 保存模型
    saveModel();

    // 导出为onnx格式
    exportModelToONNX();

    FeatureDataBase::GetInstance().updateHistoricalFeatures();

    std::cout << "=== 增量学习流程全部完成 ===" << std::endl;
}

bool IncrementalLearning::prepareTrainValSamples() {
	FeatureDataBase& feat_db = FeatureDataBase::GetInstance();

    // 加载“历史核心样本”
    // 每类历史目标取N个核心样本（N从Config读），确保覆盖旧知识
    auto historical_core_samples = feat_db.getCoreHistoricalSamples(hist_core_sample_num);
    for (const auto& [hist_obj, hist_feat] : historical_core_samples) {
        // 特征格式转换：FeatureVector→Tensor（模型只能读Tensor）
        torch::Tensor feat_tensor = ReuseFunction::GetInstance().convertFeatureToTensor(hist_feat, teacher_model_info.input_dim);
        int label_id = (int)hist_obj->toEnum(); 

        // 加入训练集（让学生模型复习旧知识）
        train_feat_batch.push_back(feat_tensor);
        train_label_batch.push_back(torch::tensor(label_id, torch::kInt64));
        // 加入历史样本验证集（后续检查是否遗忘）
        val_hist_feat.push_back(feat_tensor);
        val_hist_label.push_back(torch::tensor(label_id, torch::kInt64));
    }

    // 加载“新聚类样本”
    // 从特征库拿所有新聚类的样本，带伪标签
    auto new_clusters = feat_db.getNewCategories();
    for (const auto& cluster : new_clusters) {
        // 跳过空聚类（避免无效数据）
        if (cluster.samples.empty()) {
            continue;
        }
        // 伪标签转整数ID
        int new_label_id = convertPseudoLabelToId(cluster.pseudo_label);

        // 遍历簇内所有新样本，加入训练集和验证集
        for (const auto& new_feat : cluster.samples) {
            torch::Tensor feat_tensor = ReuseFunction::GetInstance().convertFeatureToTensor(new_feat, teacher_model_info.input_dim);
            // 加入训练集（让学生模型学新知识）
            train_feat_batch.push_back(feat_tensor);
            train_label_batch.push_back(torch::tensor(new_label_id, torch::kInt64));
            // 加入新样本验证集（后续检查是否学会）
            val_new_feat.push_back(feat_tensor);
            val_new_label.push_back(torch::tensor(new_label_id, torch::kInt64));
        }
    }

    // 检查样本是否足够（避免无数据训练）
    if (train_feat_batch.empty() || train_label_batch.empty()) {
        std::cerr << "错误：训练样本为空！请先进行新目标聚类或检查特征库" << std::endl;
        return false;
    }

    // 打印样本统计
    std::cout << "=== 样本准备完成 ===" << std::endl;
    std::cout << "总训练样本数：" << train_feat_batch.size() << std::endl;
    std::cout << "其中：历史核心样本数=" << val_hist_feat.size() << "，新聚类样本数=" << val_new_feat.size() << std::endl;
    return true;
}

bool IncrementalLearning::loadAndParseTeacherModel() {
    try {
        teacher_model = torch::jit::load(teacher_model_path);
        // 设为推理模式（仅用于提供知识，不修改它的参数）
        teacher_model.eval();

        // 解析教师模型结构
        teacher_model_info = ReuseFunction::GetInstance().analyzeModel(teacher_model_path);
        return true;

    } catch (const c10::Error& e) {
        std::cerr << "错误：加载教师模型失败！原因：" << e.what() << std::endl;
        return false;
    }
}

bool IncrementalLearning::initStudentModel(int new_class_num) {
    try {
        // 复制教师模型结构与参数
        student_model = torch::jit::load(teacher_model_path);
        // 设为训练模式
        student_model.train();

        // 扩展分类头（最后一层）适配新类别数
        // 教师模型输出维度是“历史类别数”（如5），学生要改成“历史+新类别数”（如5+2=7）
        // 学生原本只能识别5种目标，现在要加2种新目标，所以最后一层要改造成能输出7类
        // 找到最后一个全连接层（分类头，模型的“决策层”）
        std::string last_linear_layer_name;
        for (const auto& [layer_name, sub_module] : student_model.named_modules()) {
            if (sub_module.type()->name().value().name() == Config::GetInstance().getLinearType()) {
                last_linear_layer_name = layer_name; // 记录最后一个全连接层的名字
            }
        }
        // 获取教师模型分类头的权重
        torch::Tensor old_weight = student_model.attr(last_linear_layer_name + ".weight").toTensor();
        torch::Tensor old_bias = student_model.attr(last_linear_layer_name + ".bias").toTensor();
        // 新分类头的输入维度=老分类头的输入维度（特征提取部分不变）
        int new_linear_in_dim = old_weight.size(1);
        // 新分类头的输出维度=新类别数（历史+新）
        int new_linear_out_dim = new_class_num;

        // 创建新分类头的权重和偏置
        // 新权重前半部分复制老权重（保留老知识），后半部分随机初始化（学新知识）
        torch::Tensor new_weight = torch::randn({ new_linear_out_dim, new_linear_in_dim }, old_weight.options());
        new_weight.slice(0, 0, old_weight.size(0)) = old_weight; // 复制老权重
        // 新偏置同理
        torch::Tensor new_bias = torch::randn({ new_linear_out_dim }, old_bias.options());
        new_bias.slice(0, 0, old_bias.size(0)) = old_bias; // 复制老偏置

        // 替换学生模型的分类头（让学生能识别新目标）
        student_model.setattr(last_linear_layer_name + ".weight", new_weight);
        student_model.setattr(last_linear_layer_name + ".bias", new_bias);

        // 冻结特征提取层
        std::vector<torch::Tensor> trainable_params; // 存储可训练参数（仅分类头）
        for (const auto& [layer_name, sub_module] : student_model.named_modules()) {
            // 遍历层参数
            for (const auto& param : sub_module.parameters()) {
                // 如果last_linear_layer_name是一个父模块，我们只想解冻该父模块的直接参数，而不解冻其子模块的参数，那么当前代码是正确的。
                // 如果我们希望解冻整个子树，那么我们需要在last_linear_layer_name模块中递归地设置所有参数
                //for (const auto& param : sub_module.parameters(true)) { // 递归获取所有参数
                //    param.set_requires_grad(true);
                //    trainable_params.push_back(param);
                //}
                if (layer_name == last_linear_layer_name) {   
                    // 分类头：允许训练（保留梯度）
                    param.set_requires_grad(true);
                    trainable_params.push_back(param); // 加入可训练参数组

                } else {
                    // 特征提取层：冻结（禁用梯度）
                    param.set_requires_grad(false);
                }
            }
        }

        this->trainable_params = trainable_params;

        std::cout << "=== 学生模型初始化完成 ===" << std::endl;
        std::cout << "学生模型输出维度（新类别数）：" << new_linear_out_dim << std::endl;
        std::cout << "冻结所有层，仅训练最后一层分类头" << std::endl;
        return true;

    } catch (const c10::Error& e) {
        std::cerr << "错误：初始化学生模型失败！原因：" << e.what() << std::endl;
        return false;
    }
}

void IncrementalLearning::runTrainingLoop() {
    // 准备训练工具
    // 优化器：控制学生模型参数更新的“速度”（从Config读学习率）
    // 类似“学生做题的进步速度”，太快容易学错，太慢学不完
    torch::optim::Adam optimizer(
        trainable_params,
        torch::optim::AdamOptions(learning_rate)
    );
    // 分类损失函数：计算学生模型“学新目标”的错误（新样本伪标签）
    torch::nn::CrossEntropyLoss ce_loss;
    // 合并训练样本为批次（模型一次处理多个样本，效率更高）
    torch::Tensor train_feat = torch::stack(train_feat_batch);      // 特征批次（N个样本，每个2048维）
    torch::Tensor train_label = torch::stack(train_label_batch);    // 标签批次（N个样本，每个是类别ID）

    // 训练循环
    for (int epoch = 0; epoch < train_epochs; epoch++) {
        // 重置优化器的“梯度”（避免上一轮的错误影响本轮）
        optimizer.zero_grad();

        // 前向传播：计算教师和学生的输出
        // 教师模型输出（冻结参数，只给答案，不修改）
        torch::Tensor teacher_out = teacher_model.forward({ train_feat }).toTensor();
        // 学生模型输出（要学习，参数会修改）
        torch::Tensor student_out = student_model.forward({ train_feat }).toTensor();
        // 获取教师和学生的隐层特征
        torch::Tensor teacher_hidden = teacher_model.attr(Config::GetInstance().getHiddenFeat()).toTensor(); 
        torch::Tensor student_hidden = student_model.attr(Config::GetInstance().getHiddenFeat()).toTensor();

        // 计算分类损失（学生学新目标的错误）
        // 对比学生的答案（student_out）和正确答案（train_label），算差距
        // 新题的正确答案是伪标签，学生答得越近，损失越小
        torch::Tensor ce_loss_val = ce_loss(student_out, train_label);

        // 计算知识蒸馏损失（学生继承教师的旧知识）
        // 让学生的“答案分布”和“解题思路”都像教师，避免忘旧知识
        // 教师的答案是“标准答案”，学生不仅要答对，还要和教师的解题思路一样
        // 输出层蒸馏损失（KL散度）—— 让学生的答案分布像教师
        // 温度系数T：软化概率分布（让答案差异更明显，类似“放大教师的解题偏好”）
        auto softmax = torch::nn::functional::softmax; // 把输出转为概率（如[0.1,0.8,0.1]表示80%概率是第2类）
        torch::Tensor teacher_soft = softmax(teacher_out / distill_T, torch::nn::functional::SoftmaxFuncOptions(-1));
        torch::Tensor student_soft = softmax(student_out / distill_T, torch::nn::functional::SoftmaxFuncOptions(-1));
        // KL散度：计算两个概率分布的差距（越小表示学生答案越像教师）
        torch::Tensor distill_out_loss = torch::nn::functional::kl_div(
            student_soft.log(), // 学生概率的对数（KL散度公式要求）
            teacher_soft,       // 教师概率
            torch::nn::functional::KLDivFuncOptions().reduction(torch::kMean) // 求平均，避免数值过大
        ) * (distill_T * distill_T); // 温度补偿（确保损失尺度合理）

        // 隐层蒸馏损失（MSE）—— 让学生的解题思路像教师
        // MSE：计算两个隐层特征的平方差（越小表示解题思路越像教师）
        torch::Tensor distill_hid_loss = torch::nn::functional::mse_loss(student_hidden, teacher_hidden);

        // 总蒸馏损失（加权融合，λ₁+λ₂=1，从Config读）
        torch::Tensor total_distill_loss = distill_lambda1 * distill_out_loss + distill_lambda2 * distill_hid_loss;

        // 计算总损失（分类损失 + 蒸馏损失，γ从Config读）
        // 平衡“学新知识”和“不丢旧知识”，γ越大越重视旧知识
        // 总得分=新题得分（分类损失）+ 模仿教师得分（蒸馏损失）
        torch::Tensor total_loss = ce_loss_val + distill_gamma * total_distill_loss;

        // 反向传播+参数更新（学生学习进步的核心）
        // 反向传播：分析总损失的“原因”（哪些参数导致错误）
        // 参数更新：调整学生模型的参数（让下次损失更小，类似“订正错题”）
        total_loss.backward(); // 反向传播：计算参数的梯度（错误原因）
        optimizer.step();      // 参数更新：根据梯度调整参数（订正错题）

        std::cout 
            << "训练轮次 " << epoch + 1 << "/" << train_epochs
            << " | 总损失：" << total_loss.item<float>()
            << " | 新题损失（分类）：" << ce_loss_val.item<float>()
            << " | 模仿教师损失（蒸馏）：" << total_distill_loss.item<float>() 
            << std::endl;
    }

    std::cout << "=== 增量训练循环完成 ===" << std::endl;
}

bool IncrementalLearning::validateStudentModel() {
    // 设为推理模式
    student_model.eval();
    // 禁用梯度计算（验证时不修改参数，节省资源）
    torch::NoGradGuard no_grad;

    // 验证历史样本
    int hist_correct = 0; // 历史样本答对的数量
    for (size_t i = 0; i < val_hist_feat.size(); i++) {
        // 学生模型输出（对历史样本的预测答案）
        torch::Tensor pred_out = student_model.forward({ val_hist_feat[i] }).toTensor();
        // 取概率最大的类别作为预测结果
        int pred_id = torch::argmax(pred_out, 1).item<int>();
        // 真实标签
        int true_id = val_hist_label[i].item<int>();
        // 对比预测和真实，答对则计数+1
        if (pred_id == true_id) {
            hist_correct++;
        }
    }
    // 历史目标准确率 = 答对数量 / 总历史样本数
    float hist_acc = static_cast<float>(hist_correct) / val_hist_feat.size();

    // 验证新目标样本（检查是否学会新知识）
    int new_correct = 0; // 新样本答对的数量
    for (size_t i = 0; i < val_new_feat.size(); i++) {
        torch::Tensor pred_out = student_model.forward({ val_new_feat[i] }).toTensor();
        int pred_id = torch::argmax(pred_out, 1).item<int>();
        int true_id = val_new_label[i].item<int>();
        if (pred_id == true_id) {
            new_correct++;
        }
    }
    // 新目标准确率 = 答对数量 / 总新样本数
    float new_acc = static_cast<float>(new_correct) / val_new_feat.size();

    std::cout << "=== 模型验证结果 ===" << std::endl;
    std::cout << "历史目标准确率：" << hist_acc << "（要求≥" << hist_acc_thresh << "）" << std::endl;
    std::cout << "新目标准确率：" << new_acc << "（要求≥" << new_acc_thresh << "）" << std::endl;

    // 两个准确率都达标才算合格
    if (hist_acc >= hist_acc_thresh && new_acc >= new_acc_thresh) {
        std::cout << "模型验证合格！学生模型转正为新教师" << std::endl;
        return true;
    } else {
        std::cerr << "模型验证不合格！需重新训练（如调整学习率、增加样本）" << std::endl;
        return false;
    }
}

void IncrementalLearning::saveModel() {
    try {
        // 保存学生模型为新教师模型
        student_model.save(teacher_model_path);
        std::cout << "新教师模型已保存到：" << teacher_model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "错误：保存模型失败！原因：" << e.what() << std::endl;
    }
}

void IncrementalLearning::exportModelToONNX() {
    std::string cmd(
        "python " + 
        Config::GetInstance().getExportModelPyFilePath() + " \" " + 
        teacher_model_path + " \" " + 
        onnx_export_path + " \" " + 
		Config::GetInstance().getOnnxInputName() + " \" " +
		Config::GetInstance().getOnnxOutputName() + " \" " +
        std::to_string(teacher_model_info.input_dim)
    );  

    // 执行命令（返回0表示成功）
    int result = system(cmd.c_str());
    if (result != 0) {
        std::cerr << "Python脚本调用失败，命令：" << cmd << std::endl;
    } else {
		std::cout << "模型已导出为ONNX格式，路径：" << onnx_export_path << std::endl;
    }
}

int IncrementalLearning::countNewClassNum() {
    FeatureDataBase& feat_db = FeatureDataBase::GetInstance();
    auto new_clusters = feat_db.getNewCategories();
    // 用哈希表去重（避免同一新类别被多次统计）
    std::unordered_set<std::string> new_class_labels;
    for (const auto& cluster : new_clusters) {
        new_class_labels.insert(cluster.pseudo_label.label);
    }
    // 总类别数 = 历史类别数（教师模型输出维度） + 新类别数
    int total_new_class_num = teacher_model_info.output_dim + new_class_labels.size();
    std::cout 
        << "统计新类别数量：历史类别数=" 
        << teacher_model_info.output_dim
        << "，新类别数=" << new_class_labels.size()
        << "，总类别数=" << total_new_class_num 
        << std::endl;
    return total_new_class_num;
}

int IncrementalLearning::convertPseudoLabelToId(const PseudoLabel& pseudo_label) {
    // 历史类别ID范围：0~历史类别数-1（如历史5类，ID 0-4）
    int hist_class_num = teacher_model_info.output_dim;
    // 新类别ID从“历史类别数”开始递增（如5类后，新类别ID 5,6,7...）
    // 此处用伪标签的哈希值生成唯一ID（确保同一新类别ID一致）
    std::hash<std::string> hash_fn;
    int new_class_id = hist_class_num + (hash_fn(pseudo_label.label) % 100); // 限制ID范围（0-99）
    return new_class_id;
}
