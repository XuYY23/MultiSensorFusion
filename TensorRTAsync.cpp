#include "TensorRTAsync.h"

std::atomic<bool> stopInfer{ false };					// ֹͣ�����־�����������̵߳�ֹͣ

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
        return { -1, cv::Mat() }; // ���ֹͣ����������Ч֡
    }
    uint64_t fId = this->frameId.fetch_add(1); // ��ȡ��ǰ֡ID������
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
    // ���������߳�
    std::vector<std::thread> inferThreads;
    for (int i = 0; i < maxContexts; ++i) {
        inferThreads.emplace_back(&TensorRTAsync::inferThreadWorker, this, i, std::ref(frameQueue_));
    }
    // ���������߳�
    std::thread postprocessThread(&TensorRTAsync::postprocessThreadWorker, this, std::ref(frameQueue_));
    // ���������߳�
    std::thread drawThread(&TensorRTAsync::drawThreaWorker, this, std::ref(frameQueue_));
    // �ȴ������߳����
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
        return false; // �����ļ������ڻ��޷���
    }

    // ��ȡ�����ļ�
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

    // ��ȡ����/�����������
    input_tensor_name_ = engine_->getIOTensorName(0); // ��һ��Ϊ����
    output_tensor_name_ = engine_->getIOTensorName(1); // �ڶ���Ϊ���

    // ��ȡ����/����ߴ�
    nvinfer1::Dims input_dims = engine_->getTensorShape(input_tensor_name_.c_str());
    nvinfer1::Dims output_dims = engine_->getTensorShape(output_tensor_name_.c_str());

    // ���������С��NCHW: 1*3*H*W��
    input_size_ = 1;
    for (int i = 0; i < input_dims.nbDims; ++i) {
        input_size_ *= input_dims.d[i];
    }
    input_size_ *= sizeof(float);

    // ���������С
    output_size_ = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_size_ *= output_dims.d[i];
    }
    output_size_ *= sizeof(float);

    // Ԥ�ȷ����㹻���ڴ棬��ֹ�����Ŀ�������
    contexts_.reserve(maxContexts);

    // ����ִ��������
    for (int i = 0; i < maxContexts; ++i) {
        Context ctx;
        ctx.context.reset(engine_->createExecutionContext());
        if (!ctx.context) {
            std::cerr << "Failed to create execution context." << std::endl;
            return false;
        }
        // ����CUDA�����¼�
        ctx.stream.reset(new cudaStream_t);
        cudaStreamCreate(ctx.stream.get());
        ctx.event.reset(new cudaEvent_t);
        cudaEventCreateWithFlags(ctx.event.get(), cudaEventDisableTiming);
        // �����������������
        cudaMalloc(&ctx.deviceInput, input_size_);
        cudaMalloc(&ctx.deviceOutput, output_size_);
        cudaMallocHost(&ctx.hostOutput, output_size_);
		ctx.isHostOutputUseCudaMemcpy = true; // ʹ��cudaMemcpy���������������
        contexts_.push_back(std::move(ctx));
    }
}

void TensorRTAsync::preprocess(const cv::Mat& image, Context& context) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_width_, input_height_));

    // BGRתRGB����һ����[0,1]
    cv::Mat rgb, float_mat;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(float_mat, CV_32F, 1.0f / 255.0f);

    // ת��ΪNCHW��ʽ��1x3xHxW��
    std::vector<cv::Mat> channels(3);
    cv::split(float_mat, channels);

    cv::Mat input(1, input_size_ / sizeof(float), CV_32F);
    float* ptr = input.ptr<float>();
    for (int c = 0; c < 3; ++c) {
        memcpy(ptr, channels[c].data, input_height_ * input_width_ * sizeof(float));
        ptr += input_height_ * input_width_;
    }

    // ��������������״
    context.context->setInputShape(input_tensor_name_.c_str(), nvinfer1::Dims4(1, 3, input_height_, input_width_));
    // �������ݵ��豸���뻺����
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
            break; // ֹͣ�����־
        }

        // Ԥ����ͼ��
        preprocess(frameData.second, context);
        // �����������������ַ
        context.context->setTensorAddress(input_tensor_name_.c_str(), context.deviceInput);
        context.context->setTensorAddress(output_tensor_name_.c_str(), context.deviceOutput);
        // ִ���첽����
        context.context->enqueueV3(*context.stream);
        // ��¼�¼�
        cudaEventRecord(*context.event, *context.stream);
        // �ȴ��¼���ɲ�������ɶ���
        doneQueue.push({ frameData.first, contextIdx });
    }
}

void TensorRTAsync::postprocessThreadWorker(FrameQueue& frameQueue) {
    while (true) {
        auto [frameId, contextIdx] = doneQueue.pop(stopInfer);
        if (stopInfer.load() == true) {
            break; // ֹͣ�����־
        }
        // ��������������Ƿ���Ч
        if (contextIdx < 0 || contextIdx >= maxContexts) {
            std::cerr << "Invalid context index: " << contextIdx << std::endl;
            continue;
        }
        Context& context = contexts_[contextIdx];
        // �ȴ��������
        cudaEventSynchronize(*context.event);
        // ���豸����������������������
        cudaError_t cudaStatus = cudaMemcpy(context.hostOutput, context.deviceOutput, output_size_, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaMemcpy (DeviceToHost) failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            continue;
        }
        // ������
        ImageDetectionResult detectionRes;
        postprocess(frameQueue.getFrame(frameId), (float*)context.hostOutput, output_size_, detectionRes);
        // ֪ͨ����������
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
            break; // ֹͣ���Ʊ�־
        }

        // һ�������ѣ�ѭ�����Ի�����֡
        while (true) {
            auto it = drawMap.find(nextDrawId);
            if (it == drawMap.end()) {
                // û����һ֡�Ľ��������ȥ�ٵ�
                break;
            }
            // �ҵ������Ʋ�ɾ��
            render(frameQueue.getFrame(nextDrawId), it->second);
            drawMap.erase(it);
            ++nextDrawId;
        }
    }
}

void TensorRTAsync::render(const cv::Mat& image, const ImageDetectionResult& detectionRes) {
    // ���Ƽ����
    for (const auto& box : detectionRes.boxes) {
        // ���Ʊ߽��
        cv::rectangle(image, box.rect, cv::Scalar(0, 255, 0), 2);
        // ��������ǩ�����Ŷ�
        std::string label = class_names_[box.class_id] + ": " + std::to_string(box.confidence).substr(0, 4);
        // ����ǩ��ӵ��������
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);
        // �ڱ߽���Ϸ����Ʊ�ǩ����
        cv::rectangle(
            image,
            cv::Point(box.rect.x, box.rect.y - text_size.height - baseline),
            cv::Point(box.rect.x + text_size.width, box.rect.y),
            cv::Scalar(0, 255, 0),
            -1
        );
        // �ڱ߽���Ϸ����Ʊ�ǩ�ı�
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
    // ��ʾͼ��򱣴���
    cv::imshow("Detection Result", image);
    // ȷ������ˢ��
    cv::waitKey(1);
}

