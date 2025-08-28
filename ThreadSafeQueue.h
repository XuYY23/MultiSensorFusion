#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

// 线程安全队列
template<typename T>
class ThreadSafeQueue {
	std::queue<T>			m_queue;
	mutable	std::mutex		m_mtx;
	std::condition_variable m_cv;
public:
	void push(const T& value) {
		{
			std::lock_guard<std::mutex> lock(m_mtx);
			m_queue.push(value);
		}
		m_cv.notify_one();
	}

	T pop() {
		std::unique_lock<std::mutex> lock(m_mtx);
		// 等待直到队列不为空
		m_cv.wait(lock, [this] () {
				return !m_queue.empty();
			}
		);
		T value = std::move(m_queue.front());
		m_queue.pop();
		return value;
	}

	T pop(std::atomic<bool>& flag) {
		std::unique_lock<std::mutex> lock(m_mtx);
		// 等待直到队列不为空或停止推理标志被设置
		m_cv.wait(lock, [&] {
				return !m_queue.empty() || flag.load() == true;
			}
		);
		if (flag.load() == true) {
			return T();	// 如果停止推理标志被设置，返回默认值
		}
		T value = std::move(m_queue.front());
		m_queue.pop();
		return value;
	}

	bool empty() {
		std::lock_guard<std::mutex> lock(m_mtx);
		return m_queue.empty();
	}

	void wakeup() {
		m_cv.notify_all();	// 通知所有等待的线程
	}
};