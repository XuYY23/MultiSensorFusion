#pragma once

#include <mutex>

template<typename T>
class MutexValue {
	T value;
	std::mutex mtx;
public:
	void setValue(const T& v_) {
		std::lock_guard<std::mutex> lock(mtx);
		value = v_;
	}

	T getValue() {
		std::lock_guard<std::mutex> lock(mtx);
		return value;
	}
};