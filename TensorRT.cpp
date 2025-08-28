#include "TensorRT.h"
#include "TensorRTAsync.h"

TensorRT::TensorRT(const std::string& model_path, int inputW, int inputH, float conf_threshold, float nms_threshold) : 
	is_engine_loaded_(false),
	model_path_(model_path),
	input_width_(inputW),
	input_height_(inputH),
	conf_threshold_(conf_threshold), 
	nms_threshold_(nms_threshold) {
	
}

ImageDetectionResult TensorRT::detect(const cv::Mat& image) {
    if (is_engine_loaded_ == false) {
        createEngine(model_path_);
        is_engine_loaded_ = true;
    }
	preprocess(image, contexts_[0]);
    void* buffers[] = { contexts_[0].deviceInput, contexts_[0].deviceOutput };
    if (!contexts_[0].context->executeV2(buffers)) {  
        std::cerr << "Inference failed: executeV2 returned false" << std::endl;
        return ImageDetectionResult();
    }
    cudaError_t cudaStatus = cudaMemcpy(contexts_[0].hostOutput, contexts_[0].deviceOutput, output_size_, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy (DeviceToHost) failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
    ImageDetectionResult detectionRes;
    postprocess(image, (float*)contexts_[0].hostOutput, output_size_, detectionRes);
	return detectionRes;
}

void TensorRT::createEngine(const std::string& model_path) {
    std::string engine_path = model_path + ".engine";

    // 尝试加载已有引擎
    if (loadEngine(engine_path)) {
        std::cout << "Loaded engine from: " << engine_path << std::endl;
        return;
    }

    // 从ONNX构建引擎
    std::cout << "Building engine from ONNX: " << model_path << std::endl;
    buildEngine(model_path, engine_path);

    // 再次加载构建好的引擎
    if (!loadEngine(engine_path)) {
        throw std::runtime_error("Failed to load built engine");
    }
}

bool TensorRT::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        return false;
    }

    // 读取引擎文件
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // 创建运行时
    runtime_ = TRTUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    if (!runtime_) {
        return false;
    }

    // 反序列化引擎
    engine_ = TRTUniquePtr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(engine_data.data(), size)
    );
    if (!engine_) {
        return false;
    }

    // 创建执行上下文
    Context oneContext;
    oneContext.context = TRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!oneContext.context) {
        return false;
    }

    // 获取输入/输出张量名称（TensorRT 10.x需显式获取）
    input_tensor_name_ = engine_->getIOTensorName(0); // 第一个为输入
    output_tensor_name_ = engine_->getIOTensorName(1); // 第二个为输出

    // 获取输入/输出尺寸
    nvinfer1::Dims input_dims = engine_->getTensorShape(input_tensor_name_.c_str());
    nvinfer1::Dims output_dims = engine_->getTensorShape(output_tensor_name_.c_str());

    // 计算输入大小（NCHW: 1*3*H*W）
    input_size_ = 1;
    for (int i = 0; i < input_dims.nbDims; ++i) {
        input_size_ *= input_dims.d[i];
    }
    input_size_ *= sizeof(float);

    // 计算输出大小
    output_size_ = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_size_ *= output_dims.d[i];
    }
    output_size_ *= sizeof(float);

    oneContext.hostOutput = new float[(float)output_size_ / sizeof(float)];
	oneContext.isHostOutputUseCudaMemcpy = false; 

    // 分配GPU内存
    cudaMalloc(&oneContext.deviceInput, input_size_);
    cudaMalloc(&oneContext.deviceOutput, output_size_);

	contexts_.push_back(std::move(oneContext));

    return true;
}

void TensorRT::buildEngine(const std::string& model_path, const std::string& engine_path) {
    // 创建构建器
    TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        throw std::runtime_error("Failed to create builder");
    }

    // 创建网络（显式批次）
    const unsigned explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicit_batch));
    if (!network) {
        throw std::runtime_error("Failed to create network");
    }

    // 解析ONNX
    TRTUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger));
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR))) {
        throw std::runtime_error("Failed to parse ONNX: " + model_path);
    }

    // 配置构建器
    TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) {
        throw std::runtime_error("Failed to create builder config");
    }

    // 设置工作区大小（TensorRT 10.x使用setMemoryPoolLimit）
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB

    // 启用FP16加速
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // 构建序列化引擎
    TRTUniquePtr<nvinfer1::IHostMemory> serialized_engine(builder->buildSerializedNetwork(*network, *config));
    if (!serialized_engine) {
        throw std::runtime_error("Failed to build engine");
    }

    // 保存引擎
    saveEngine(engine_path, serialized_engine.get());
}

void TensorRT::saveEngine(const std::string& engine_path, nvinfer1::IHostMemory* serialized_engine) {
    std::ofstream file(engine_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open engine file: " + engine_path);
    }
    file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
}

void TensorRT::preprocess(const cv::Mat& image, Context& context) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_width_, input_height_));

    // BGR转RGB，归一化到[0,1]
    cv::Mat rgb, float_mat;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(float_mat, CV_32F, 1.0f / 255.0f);

    // 转换为NCHW格式（1x3xHxW）
    std::vector<cv::Mat> channels(3);
    cv::split(float_mat, channels);

    cv::Mat input(1, input_size_ / sizeof(float), CV_32F);
    float* ptr = input.ptr<float>();
    for (int c = 0; c < 3; ++c) {
        memcpy(ptr, channels[c].data, input_height_ * input_width_ * sizeof(float));
        ptr += input_height_ * input_width_;
    }

    // 设置输入张量形状
    context.context->setInputShape(input_tensor_name_.c_str(), nvinfer1::Dims4(1, 3, input_height_, input_width_));
    // 拷贝数据到设备输入缓冲区
    cudaError_t cudaStatus = cudaMemcpy(context.deviceInput, input.data, input_size_, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy (HostToDevice) failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }
}

void TensorRT::postprocess(const cv::Mat& image, const float* output, int output_size, ImageDetectionResult& detectionRes) {
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    // 动态获取类别数量
    int num_classes = class_names_.size();

    // YOLOv8 输出格式：[8400, 85] = [x,y,w,h,conf,cls0...cls79]
    for (int i = 0; i < output_size; i += 85) {
        float conf = output[i + 4];
        if (conf < conf_threshold_) {
            continue;
        }

        // 查找最大类别置信度和类别 ID
        int cls_id = 0;
        float max_cls_conf = 0.0f;
        for (int j = 0; j < num_classes; ++j) { // 动态获取类别数量
            if (output[i + 5 + j] > max_cls_conf) {
                max_cls_conf = output[i + 5 + j];
                cls_id = j;
            }
        }

        float final_conf = conf * max_cls_conf;
        if (final_conf < conf_threshold_) {
            continue;
        }

        // 边界框坐标转换回原始图像尺寸
        float cx = output[i] * image.cols;
        float cy = output[i + 1] * image.rows;
        float w = output[i + 2] * image.cols;
        float h = output[i + 3] * image.rows;

        int x = std::max(0, static_cast<int>(cx - w / 2));
        int y = std::max(0, static_cast<int>(cy - h / 2));
        int width = std::min(static_cast<int>(w), image.cols - x);
        int height = std::min(static_cast<int>(h), image.rows - y);

        width = std::max(0, width);
        height = std::max(0, height);

        boxes.emplace_back(x, y, width, height);
        confidences.push_back(final_conf);
        class_ids.push_back(cls_id);
    }

    // NMS 非极大值抑制
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);

    for (int idx : indices) {
        if (class_ids[idx] < 0 || class_ids[idx] >= static_cast<int>(class_names_.size())) {
            std::cerr << "Class index out of range: " << class_ids[idx] << std::endl;
            continue;
        }
        ImageDetectionResult::Box box;
        box.rect = boxes[idx];
        box.class_id = class_ids[idx];
        box.confidence = confidences[idx];
        detectionRes.boxes.push_back(box);
    }
}
