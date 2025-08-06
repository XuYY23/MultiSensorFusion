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

// 使用nlohmann/json库处理JSON
using json = nlohmann::json;

// 时间戳类型定义，使用系统时钟
using Timestamp = std::chrono::system_clock::time_point;

template<typename T>
using TRTUniquePtr = std::unique_ptr<T>;