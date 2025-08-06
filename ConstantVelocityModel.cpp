#include "ConstantVelocityModel.h"

ConstantVelocityModel::ConstantVelocityModel() : 
    position_process_noise_(0.1),
    velocity_process_noise_(0.05) {
}

// 预测目标在下一时刻的状态
FusedObject ConstantVelocityModel::predict(const FusedObject& object, Timestamp current_time, Timestamp future_time) {
    FusedObject predicted = object;
    predicted.timestamp = future_time;

    // 计算时间差（秒）
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(future_time - current_time);
	double dt = duration.count() / 1e6; // 转换为秒

    // 状态转移矩阵 [1  0  0  dt 0  0;
    //              0  1  0  0  dt 0;
    //              0  0  1  0  0  dt;
    //              0  0  0  1  0  0;
    //              0  0  0  0  1  0;
    //              0  0  0  0  0  1]
	Eigen::MatrixXd F(6, 6); 
    F.setIdentity();
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;

    // 状态向量 [x, y, z, vx, vy, vz]
    Eigen::VectorXd x(6);
    x << object.position.x(), object.position.y(), object.position.z(),
         object.velocity.x(), object.velocity.y(), object.velocity.z();

    // 预测状态
    Eigen::VectorXd x_pred = F * x;

    // 更新位置和速度
    predicted.position = Eigen::Vector3d(x_pred[0], x_pred[1], x_pred[2]);
    predicted.velocity = Eigen::Vector3d(x_pred[3], x_pred[4], x_pred[5]);

    // 过程噪声矩阵
    Eigen::MatrixXd Q(6, 6);
    Q.setZero();
    Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * position_process_noise_ * dt;
    Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * velocity_process_noise_;

    // 预测协方差
    Eigen::MatrixXd P(6, 6);
    P.block<3, 3>(0, 0) = object.position_covariance;
    P.block<3, 3>(3, 3) = object.velocity_covariance;

    Eigen::MatrixXd P_pred = F * P * F.transpose() + Q;

    // 更新协方差
    predicted.position_covariance = P_pred.block<3, 3>(0, 0);
    predicted.velocity_covariance = P_pred.block<3, 3>(3, 3);

    return predicted;
}

// 更新目标状态
FusedObject ConstantVelocityModel::update(const FusedObject& predicted_object, const Detection& detection) {
    FusedObject updated = predicted_object;

    // 测量矩阵[ 1  0  0  0  0  0;
    //          0  1  0  0  0  0;
    //          0  0  1  0  0  0;
    //          0  0  0  1  0  0;
    //          0  0  0  0  1  0;
    //          0  0  0  0  0  1]
    Eigen::MatrixXd H(6, 6);
    H.setIdentity();

    // 状态向量 [x, y, z, vx, vy, vz]
    Eigen::VectorXd x_pred(6);
    x_pred << predicted_object.position.x(), predicted_object.position.y(), predicted_object.position.z(),
              predicted_object.velocity.x(), predicted_object.velocity.y(), predicted_object.velocity.z();

    // 测量向量
    Eigen::VectorXd z(6);
    z << detection.position_global.x(), detection.position_global.y(), detection.position_global.z(),
         detection.velocity_global.x(), detection.velocity_global.y(), detection.velocity_global.z();

    // 测量噪声协方差
    Eigen::MatrixXd R(6, 6);
    R.setZero();
    R.block<3, 3>(0, 0) = detection.sensor_type == SensorType::RADAR ? detection.covariance * 2.0 : detection.covariance;   // 雷达测量噪声稍大
    R.block<3, 3>(3, 3) = R.block<3, 3>(0, 0) * 0.5;  // 速度噪声通常较小

    // 计算残差
    Eigen::VectorXd y = z - H * x_pred;

    // 计算卡尔曼增益
    Eigen::MatrixXd P_pred(6, 6);
    P_pred.block<3, 3>(0, 0) = predicted_object.position_covariance;
    P_pred.block<3, 3>(3, 3) = predicted_object.velocity_covariance;

	Eigen::MatrixXd S = H * P_pred * H.transpose() + R;         // 残差协方差
	Eigen::MatrixXd K = P_pred * H.transpose() * S.inverse();   // 卡尔曼增益

    // 更新状态
    Eigen::VectorXd x_updated = x_pred + K * y;
    updated.position = Eigen::Vector3d(x_updated[0], x_updated[1], x_updated[2]);
    updated.velocity = Eigen::Vector3d(x_updated[3], x_updated[4], x_updated[5]);

    // 更新协方差
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