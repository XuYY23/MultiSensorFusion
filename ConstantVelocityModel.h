#pragma once

#include "structs.h"

// �㶨�ٶ��˶�ģ�ͣ�����Ŀ��״̬Ԥ��͸���
class ConstantVelocityModel {
public:
    ConstantVelocityModel();
    
    // Ԥ��Ŀ������һʱ�̵�״̬
    FusedObject predict(const FusedObject& object, Timestamp current_time, Timestamp future_time);
    
    // ����Ŀ��״̬
    FusedObject update(const FusedObject& predicted_object, const Detection& detection);
    
    // ���ù�������
    void setProcessNoise(double position_noise, double velocity_noise);
    
private:
    double position_process_noise_;  // λ�ù�������
    double velocity_process_noise_;  // �ٶȹ�������
};