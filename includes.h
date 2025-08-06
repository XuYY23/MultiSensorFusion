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

// ʹ��nlohmann/json�⴦��JSON
using json = nlohmann::json;

// ʱ������Ͷ��壬ʹ��ϵͳʱ��
using Timestamp = std::chrono::system_clock::time_point;