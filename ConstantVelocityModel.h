#pragma once

#include "structs.h"

// 恒定速度运动模型，用于目标状态预测和更新
class ConstantVelocityModel {
public:
    ConstantVelocityModel();
    
    // 预测目标在下一时刻的状态
    FusedObject predict(const FusedObject& object, Timestamp current_time, Timestamp future_time);
    
    // 更新目标状态
    FusedObject update(const FusedObject& predicted_object, const Detection& detection);
    
    // 设置过程噪声
    void setProcessNoise(double position_noise, double velocity_noise);
    
private:
    double position_process_noise_;  // 位置过程噪声
    double velocity_process_noise_;  // 速度过程噪声
};