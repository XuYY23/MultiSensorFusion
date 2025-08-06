#include "ConstantVelocityModel.h"

ConstantVelocityModel::ConstantVelocityModel() : 
    position_process_noise_(0.1),
    velocity_process_noise_(0.05) {
}

// Ԥ��Ŀ������һʱ�̵�״̬
FusedObject ConstantVelocityModel::predict(const FusedObject& object, Timestamp current_time, Timestamp future_time) {
    FusedObject predicted = object;
    predicted.timestamp = future_time;

    // ����ʱ���룩
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(future_time - current_time);
	double dt = duration.count() / 1e6; // ת��Ϊ��

    // ״̬ת�ƾ��� [1  0  0  dt 0  0;
    //              0  1  0  0  dt 0;
    //              0  0  1  0  0  dt;
    //              0  0  0  1  0  0;
    //              0  0  0  0  1  0;
    //              0  0  0  0  0  1]
	Eigen::MatrixXd F(6, 6); 
    F.setIdentity();
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;

    // ״̬���� [x, y, z, vx, vy, vz]
    Eigen::VectorXd x(6);
    x << object.position.x(), object.position.y(), object.position.z(),
         object.velocity.x(), object.velocity.y(), object.velocity.z();

    // Ԥ��״̬
    Eigen::VectorXd x_pred = F * x;

    // ����λ�ú��ٶ�
    predicted.position = Eigen::Vector3d(x_pred[0], x_pred[1], x_pred[2]);
    predicted.velocity = Eigen::Vector3d(x_pred[3], x_pred[4], x_pred[5]);

    // ������������
    Eigen::MatrixXd Q(6, 6);
    Q.setZero();
    Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * position_process_noise_ * dt;
    Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * velocity_process_noise_;

    // Ԥ��Э����
    Eigen::MatrixXd P(6, 6);
    P.block<3, 3>(0, 0) = object.position_covariance;
    P.block<3, 3>(3, 3) = object.velocity_covariance;

    Eigen::MatrixXd P_pred = F * P * F.transpose() + Q;

    // ����Э����
    predicted.position_covariance = P_pred.block<3, 3>(0, 0);
    predicted.velocity_covariance = P_pred.block<3, 3>(3, 3);

    return predicted;
}

// ����Ŀ��״̬
FusedObject ConstantVelocityModel::update(const FusedObject& predicted_object, const Detection& detection) {
    FusedObject updated = predicted_object;

    // ��������[ 1  0  0  0  0  0;
    //          0  1  0  0  0  0;
    //          0  0  1  0  0  0;
    //          0  0  0  1  0  0;
    //          0  0  0  0  1  0;
    //          0  0  0  0  0  1]
    Eigen::MatrixXd H(6, 6);
    H.setIdentity();

    // ״̬���� [x, y, z, vx, vy, vz]
    Eigen::VectorXd x_pred(6);
    x_pred << predicted_object.position.x(), predicted_object.position.y(), predicted_object.position.z(),
              predicted_object.velocity.x(), predicted_object.velocity.y(), predicted_object.velocity.z();

    // ��������
    Eigen::VectorXd z(6);
    z << detection.position_global.x(), detection.position_global.y(), detection.position_global.z(),
         detection.velocity_global.x(), detection.velocity_global.y(), detection.velocity_global.z();

    // ��������Э����
    Eigen::MatrixXd R(6, 6);
    R.setZero();
    R.block<3, 3>(0, 0) = detection.sensor_type == SensorType::RADAR ? detection.covariance * 2.0 : detection.covariance;   // �״���������Դ�
    R.block<3, 3>(3, 3) = R.block<3, 3>(0, 0) * 0.5;  // �ٶ�����ͨ����С

    // ����в�
    Eigen::VectorXd y = z - H * x_pred;

    // ���㿨��������
    Eigen::MatrixXd P_pred(6, 6);
    P_pred.block<3, 3>(0, 0) = predicted_object.position_covariance;
    P_pred.block<3, 3>(3, 3) = predicted_object.velocity_covariance;

	Eigen::MatrixXd S = H * P_pred * H.transpose() + R;         // �в�Э����
	Eigen::MatrixXd K = P_pred * H.transpose() * S.inverse();   // ����������

    // ����״̬
    Eigen::VectorXd x_updated = x_pred + K * y;
    updated.position = Eigen::Vector3d(x_updated[0], x_updated[1], x_updated[2]);
    updated.velocity = Eigen::Vector3d(x_updated[3], x_updated[4], x_updated[5]);

    // ����Э����
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(6, 6);
    Eigen::MatrixXd P_updated = (I - K * H) * P_pred;
    updated.position_covariance = P_updated.block<3, 3>(0, 0);
    updated.velocity_covariance = P_updated.block<3, 3>(3, 3);

    return updated;
}

void ConstantVelocityModel::setProcessNoise(double position_noise, double velocity_noise) {
    position_process_noise_ = position_noise;
    velocity_process_noise_ = velocity_noise;
}