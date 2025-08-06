#pragma once

#include "ThreadSafeQueue.h"
#include "TensorRT.h"

extern std::atomic<bool> stopInfer;					// 停止推理标志，控制推理线程的停止

// 帧队列，存储每一帧图像数据
class FrameQueue {
	std::atomic<uint64_t> frameId{ 0 };				// 每一帧的唯一标识
	std::queue<cv::Mat> frameQueue;					// 帧队列
	mutable std::mutex queueMutex;					// 互斥锁，保护队列的线程安全
	std::condition_variable queueCv;				// 条件变量，用于通知有新帧加入队列
	std::unordered_map<uint64_t, cv::Mat> frameMap;	// 存储帧ID和对应的图像数据
public:
	void push(const cv::Mat& frame);
	// 获取队列中的一帧图像和对应的帧ID
	std::pair<uint64_t, cv::Mat> pop();
	cv::Mat getFrame(uint64_t fId);
	bool empty();
	uint64_t frameCount() { return frameId.load(); }
	void wakeup() {
		queueCv.notify_all();	// 通知所有等待的线程
	}
};

class TensorRTAsync : public TensorRT {
	int maxContexts = 3;										// 最大上下文数量
	ThreadSafeQueue<std::pair<uint64_t, int>> doneQueue;		// 完成队列，存储已处理的帧ID和应用的是哪一个Context编号
	std::unordered_map<uint64_t, ImageDetectionResult> drawMap;	// 待绘制的检测结果，存储帧ID和对应的检测结果
	uint64_t nextDrawId = 0;									// 下一个待绘制的帧ID
	std::mutex drawMutex;										// 绘制结果的互斥锁
	std::condition_variable drawCv;								// 绘制结果的条件变量

	FrameQueue& frameQueue_;										// 帧队列

public:
	TensorRTAsync(FrameQueue& frameQueue, const std::string& model_path, int input_w, int input_h, float conf_thresh = 0.5f, float iou_thresh = 0.45f);
	ImageDetectionResult detect(const cv::Mat& image) override;
	void wakeup() {
		drawCv.notify_all();	// 通知所有等待的线程
		doneQueue.wakeup();
		std::cout << "TargetDetectTensorRTAsync wakeup done." << std::endl;
	}

	uint64_t getNextDrawId() {
		std::lock_guard<std::mutex> lock(drawMutex);
		return nextDrawId;
	}

private:
	bool loadEngine(const std::string& engine_path) override;
	// 预处理
	void preprocess(const cv::Mat& image, Context& context) override;
	//推理工作线程函数
	void inferThreadWorker(int contextIdx, FrameQueue& frameQueue);
	// 后处理工作线程函数
	void postprocessThreadWorker(FrameQueue& frameQueue);
	// 通知帧处理完成
	void notifyFrameDone(const uint64_t& frameId, const ImageDetectionResult& detectionRes);
	// 绘制线程函数
	void drawThreaWorker(FrameQueue& frameQueue);
	void render(const cv::Mat& image, const ImageDetectionResult& detectionRes);
};