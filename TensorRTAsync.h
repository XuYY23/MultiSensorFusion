#pragma once

#include "ThreadSafeQueue.h"
#include "TensorRT.h"

extern std::atomic<bool> stopInfer;					// ֹͣ�����־�����������̵߳�ֹͣ

// ֡���У��洢ÿһ֡ͼ������
class FrameQueue {
	std::atomic<uint64_t> frameId{ 0 };				// ÿһ֡��Ψһ��ʶ
	std::queue<cv::Mat> frameQueue;					// ֡����
	mutable std::mutex queueMutex;					// ���������������е��̰߳�ȫ
	std::condition_variable queueCv;				// ��������������֪ͨ����֡�������
	std::unordered_map<uint64_t, cv::Mat> frameMap;	// �洢֡ID�Ͷ�Ӧ��ͼ������
public:
	void push(const cv::Mat& frame);
	// ��ȡ�����е�һ֡ͼ��Ͷ�Ӧ��֡ID
	std::pair<uint64_t, cv::Mat> pop();
	cv::Mat getFrame(uint64_t fId);
	bool empty();
	uint64_t frameCount() { return frameId.load(); }
	void wakeup() {
		queueCv.notify_all();	// ֪ͨ���еȴ����߳�
	}
};

class TensorRTAsync : public TensorRT {
	int maxContexts = 3;										// �������������
	ThreadSafeQueue<std::pair<uint64_t, int>> doneQueue;		// ��ɶ��У��洢�Ѵ����֡ID��Ӧ�õ�����һ��Context���
	std::unordered_map<uint64_t, ImageDetectionResult> drawMap;	// �����Ƶļ�������洢֡ID�Ͷ�Ӧ�ļ����
	uint64_t nextDrawId = 0;									// ��һ�������Ƶ�֡ID
	std::mutex drawMutex;										// ���ƽ���Ļ�����
	std::condition_variable drawCv;								// ���ƽ������������

	FrameQueue& frameQueue_;										// ֡����

public:
	TensorRTAsync(FrameQueue& frameQueue, const std::string& model_path, int input_w, int input_h, float conf_thresh = 0.5f, float iou_thresh = 0.45f);
	ImageDetectionResult detect(const cv::Mat& image) override;
	void wakeup() {
		drawCv.notify_all();	// ֪ͨ���еȴ����߳�
		doneQueue.wakeup();
		std::cout << "TargetDetectTensorRTAsync wakeup done." << std::endl;
	}

	uint64_t getNextDrawId() {
		std::lock_guard<std::mutex> lock(drawMutex);
		return nextDrawId;
	}

private:
	bool loadEngine(const std::string& engine_path) override;
	// Ԥ����
	void preprocess(const cv::Mat& image, Context& context) override;
	//�������̺߳���
	void inferThreadWorker(int contextIdx, FrameQueue& frameQueue);
	// �������̺߳���
	void postprocessThreadWorker(FrameQueue& frameQueue);
	// ֪ͨ֡�������
	void notifyFrameDone(const uint64_t& frameId, const ImageDetectionResult& detectionRes);
	// �����̺߳���
	void drawThreaWorker(FrameQueue& frameQueue);
	void render(const cv::Mat& image, const ImageDetectionResult& detectionRes);
};