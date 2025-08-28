#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

// �̰߳�ȫ����
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
		// �ȴ�ֱ�����в�Ϊ��
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
		// �ȴ�ֱ�����в�Ϊ�ջ�ֹͣ�����־������
		m_cv.wait(lock, [&] {
				return !m_queue.empty() || flag.load() == true;
			}
		);
		if (flag.load() == true) {
			return T();	// ���ֹͣ�����־�����ã�����Ĭ��ֵ
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
		m_cv.notify_all();	// ֪ͨ���еȴ����߳�
	}
};