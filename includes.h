#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <chrono>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <dlib/matrix.h>
#include <dlib/optimization/max_cost_assignment.h>
#include "json.hpp"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

// ʹ��nlohmann/json�⴦��JSON
using json = nlohmann::json;

// ʱ������Ͷ��壬ʹ��ϵͳʱ��
using Timestamp = std::chrono::system_clock::time_point;

template<typename T>
using TRTUniquePtr = std::unique_ptr<T>;

#define PROPERTY(Type, Name, Func) \
private: \
    Type m_##Name; \
public: \
    const Type& get##Func() { return m_##Name; } \
    void set##Func(const Type& value) { m_##Name = value; }