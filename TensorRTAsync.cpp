#include "TensorRTAsync.h"

std::atomic<bool> stopInfer{ false };					// 停止推理标志，控制推理线程的停止

void FrameQueue::push(const cv::Mat& frame) {
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        frameQueue.push(frame.clone());
    }
    queueCv.notify_one();
}

std::pair<uint64_t, cv::Mat> FrameQueue::pop() {
    std::unique_lock<std::mutex> lock(queueMutex);
    queueCv.wait(lock, [this] {
            return !frameQueue.empty() || stopInfer.load() == true;
        }
    );
    if (stopInfer.load() == true) {
        return { -1, cv::Mat() }; // 如果停止推理，返回无效帧
    }
    uint64_t fId = this->frameId.fetch_add(1); // 获取当前帧ID并自增
    frameMap[fId] = frameQueue.front().clone();
    frameQueue.pop();
    return { fId, frameMap[fId] };
}

cv::Mat FrameQueue::getFrame(uint64_t fId) {
    std::lock_guard<std::mutex> lock(queueMutex);
    return frameMap[fId];
}

bool FrameQueue::empty() {
    std::lock_guard<std::mutex> lock(queueMutex);
    return frameQueue.empty();
}

TensorRTAsync::TensorRTAsync(FrameQueue& frameQueue, const std::string& model_path, int input_w, int input_h, float conf_thresh, float iou_thresh) :
    TensorRT(model_path, input_w, input_h, conf_thresh, iou_thresh),
    frameQueue_(std::ref(frameQueue)) {
    
}

ImageDetectionResult TensorRTAsync::detect(const cv::Mat& image) {
    // 启动推理线程
    std::vector<std::thread> inferThreads;
    for (int i = 0; i < maxContexts; ++i) {
        inferThreads.emplace_back(&TensorRTAsync::inferThreadWorker, this, i, std::ref(frameQueue_));
    }
    // 启动后处理线程
    std::thread postprocessThread(&TensorRTAsync::postprocessThreadWorker, this, std::ref(frameQueue_));
    // 启动绘制线程
    std::thread drawThread(&TensorRTAsync::drawThreaWorker, this, std::ref(frameQueue_));
    // 等待所有线程完成
    for (auto& thread : inferThreads) {
        thread.join();
    }
    postprocessThread.join();
    drawThread.join();
    return ImageDetectionResult();
}

bool TensorRTAsync::loadEngine(const std::string& engine_path) {
    std::ifstream engineFile(engine_path, std::ios::binary);
    if (!engineFile.good()) {
        return false; // 引擎文件不存在或无法打开
    }

    // 读取引擎文件
    engineFile.seekg(0, std::ios::end);
    size_t file_size = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    std::vector<char> engineData(file_size);
    engineFile.read(engineData.data(), file_size);
    engineFile.close();

    runtime_ = TRTUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    if (runtime_ == nullptr) {
        std::cerr << "Failed to create TensorRT runtime." << std::endl;
        return false;
    }

    engine_ = TRTUniquePtr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engineData.data(), file_size));
    if (engine_ == nullptr) {
        std::cerr << "Failed to deserialize TensorRT engine." << std::endl;
        return false;
    }

    // 获取输入/输出张量名称
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

    // 预先分配足够的内存，防止后续的拷贝工作
    contexts_.reserve(maxContexts);

    // 创建执行上下文
    for (int i = 0; i < maxContexts; ++i) {
        Context ctx;
        ctx.context.reset(engine_->createExecutionContext());
        if (!ctx.context) {
            std::cerr << "Failed to create execution context." << std::endl;
            return false;
        }
        // 创建CUDA流和事件
        ctx.stream.reset(new cudaStream_t);
        cudaStreamCreate(ctx.stream.get());
        ctx.event.reset(new cudaEvent_t);
        cudaEventCreateWithFlags(ctx.event.get(), cudaEventDisableTiming);
        // 分配输入输出缓冲区
        cudaMalloc(&ctx.deviceInput, input_size_);
        cudaMalloc(&ctx.deviceOutput, output_size_);
        cudaMallocHost(&ctx.hostOutput, output_size_);
		ctx.isHostOutputUseCudaMemcpy = true; // 使用cudaMemcpy分配主机输出数据
        contexts_.push_back(std::move(ctx));
    }
}

void TensorRTAsync::preprocess(const cv::Mat& image, Context& context) {
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
    cudaError_t cudaStatus = cudaMemcpyAsync(context.deviceInput, input.data, input_size_, cudaMemcpyHostToDevice, *context.stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpyAsync (HostToDevice) failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }
}

void TensorRTAsync::inferThreadWorker(int contextIdx, FrameQueue& frameQueue) {
    if (contextIdx < 0 || contextIdx >= maxContexts) {
        std::cerr << "Invalid context index: " << contextIdx << std::endl;
        return;
    }
    Context& context = contexts_[contextIdx];
    while (stopInfer.load() == false) {
        std::pair<uint64_t, cv::Mat> frameData = frameQueue.pop();
        if (frameData.first == -1) {
            break; // 停止推理标志
        }

        // 预处理图像
        preprocess(frameData.second, context);
        // 设置输入输出张量地址
        context.context->setTensorAddress(input_tensor_name_.c_str(), context.deviceInput);
        context.context->setTensorAddress(output_tensor_name_.c_str(), context.deviceOutput);
        // 执行异步推理
        context.context->enqueueV3(*context.stream);
        // 记录事件
        cudaEventRecord(*context.event, *context.stream);
        // 等待事件完成并加入完成队列
        doneQueue.push({ frameData.first, contextIdx });
    }
}

void TensorRTAsync::postprocessThreadWorker(FrameQueue& frameQueue) {
    while (true) {
        auto [frameId, contextIdx] = doneQueue.pop(stopInfer);
        if (stopInfer.load() == true) {
            break; // 停止推理标志
        }
        // 检查上下文索引是否有效
        if (contextIdx < 0 || contextIdx >= maxContexts) {
            std::cerr << "Invalid context index: " << contextIdx << std::endl;
            continue;
        }
        Context& context = contexts_[contextIdx];
        // 等待推理完成
        cudaEventSynchronize(*context.event);
        // 从设备输出缓冲区拷贝结果到主机
        cudaError_t cudaStatus = cudaMemcpy(context.hostOutput, context.deviceOutput, output_size_, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpy (DeviceToHost) failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            continue;
        }
        // 后处理结果
        ImageDetectionResult detectionRes;
        postprocess(frameQueue.getFrame(frameId), (float*)context.hostOutput, output_size_, detectionRes);
        // 通知结果处理完成
        notifyFrameDone(frameId, detectionRes);
    }
}

void TensorRTAsync::notifyFrameDone(const uint64_t& frameId, const ImageDetectionResult& detectionRes) {
    {
        std::lock_guard<std::mutex> lock(drawMutex);
        drawMap[frameId] = detectionRes;
    }
    drawCv.notify_one();
}

void TensorRTAsync::drawThreaWorker(FrameQueue& frameQueue) {
    std::unique_lock<std::mutex> lock(drawMutex);
    while (true) {
        drawCv.wait(lock, [this] {
            return (!drawMap.empty() && drawMap.find(nextDrawId) != drawMap.end()) || stopInfer.load() == true;
            });
        if (stopInfer.load() == true) {
            break; // 停止绘制标志
        }

        // 一旦被唤醒，循环尝试画连续帧
        while (true) {
            auto it = drawMap.find(nextDrawId);
            if (it == drawMap.end()) {
                // 没有下一帧的结果，跳出去再等
                break;
            }
            // 找到：绘制并删除
            render(frameQueue.getFrame(nextDrawId), it->second);
            drawMap.erase(it);
            ++nextDrawId;
        }
    }
}

void TensorRTAsync::render(const cv::Mat& image, const ImageDetectionResult& detectionRes) {
    // 绘制检测结果
    for (const auto& box : detectionRes.boxes) {
        // 绘制边界框
        cv::rectangle(image, box.rect, cv::Scalar(0, 255, 0), 2);
        // 绘制类别标签和置信度
        std::string label = class_names_[box.class_id] + ": " + std::to_string(box.confidence).substr(0, 4);
        // 将标签添加到检测结果中
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);
        // 在边界框上方绘制标签背景
        cv::rectangle(
            image,
            cv::Point(box.rect.x, box.rect.y - text_size.height - baseline),
            cv::Point(box.rect.x + text_size.width, box.rect.y),
            cv::Scalar(0, 255, 0),
            -1
        );
        // 在边界框上方绘制标签文本
        cv::putText(
            image,
            label,
            cv::Point(box.rect.x, box.rect.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 0),
            2
        );
    }
    // 显示图像或保存结果
    cv::imshow("Detection Result", image);
    // 确保窗口刷新
    cv::waitKey(1);
}

