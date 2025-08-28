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
    Type Name; \
public: \
    const Type& get##Func() { return this->Name; } \
    void set##Func(const Type& value) { this->Name = value; }

// ��Ҫ�ṩvoid init()�������ڳ�ʼ��
#define INSTANCE(class_name) \
private: \
    class_name() { init(); }; \
    class_name(const class_name&) = default; \
    class_name& operator=(const class_name&) = default; \
public: \
    static class_name& GetInstance() { \
        static class_name instance; \
        return instance; \
    }