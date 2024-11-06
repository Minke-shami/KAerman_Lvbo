//
//  main.cpp
//  卡尔曼滤波_定位
//
//  Created by mac on 2024/10/30.
//

#include <iostream>
#include <cmath>
#include <Eigen/Dense>

const double PI = 3.14159265358979323846;

struct KalmanFilter {
    Eigen::VectorXd state; // 状态向量 [x, y, theta, v]
    Eigen::MatrixXd covariance;
    Eigen::MatrixXd A;
    Eigen::MatrixXd B;
    Eigen::MatrixXd C;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;

    KalmanFilter() {
        state = Eigen::VectorXd(4);
        state << 0, 0, 0, 0;

        covariance = Eigen::MatrixXd(4, 4);
        covariance << 0.01, 0,    0,    0,
                      0,    0.01, 0,    0,
                      0,    0,    0.01, 0,
                      0,    0,    0,    0.1;

        A = Eigen::MatrixXd(4, 4);
        B = Eigen::MatrixXd(4, 2);
        B << 0,   0,
             0,   0,
             0.1, 0,
             0,   0.1;

        C = Eigen::MatrixXd(5, 4);
        C << 0.5, 0.5, 0.5, 0.5,
             0,   0,   0.1, 0.1,
             0,   0,   0,   0,
             0,   0,   0,   0,
             0,   0,   0,   0;

        Q = Eigen::MatrixXd(4, 4);
        Q << 0.01, 0,    0,    0,
             0,    0.01, 0,    0,
             0,    0,    0.01, 0,
             0,    0,    0,    0.01;

        R = Eigen::MatrixXd(5, 5);
        R << 0.01, 0,    0,    0,    0,
             0,    0.01, 0,    0,    0,
             0,    0,    0.01, 0,    0,
             0,    0,    0,    0.01, 0,
             0,    0,    0,    0,    0.01;
    }

    void predict(double deltaT, double steeringAngleSpeed, double velocityChange) {
        state[2] += steeringAngleSpeed * deltaT;
        A << 1, 0, 0, deltaT * cos(state[2]),
             0, 1, 0, deltaT * sin(state[2]),
             0, 0, 1, 0,
             0, 0, 0, 1;

        Eigen::VectorXd u(2);
        u << steeringAngleSpeed, velocityChange;

        state = A * state + B * u;
        covariance = A * covariance * A.transpose() + Q;
    }

    void update(double wheel1Speed, double wheel2Speed, double wheel3Speed, double wheel4Speed, double steeringAngle) {
        Eigen::VectorXd measurement(5);
        measurement << wheel1Speed, wheel2Speed, wheel3Speed, wheel4Speed, steeringAngle;

        Eigen::MatrixXd S = C * covariance * C.transpose() + R;
        Eigen::MatrixXd K = covariance * C.transpose() * S.inverse();
        Eigen::VectorXd innovation = measurement - C * state;

        state = state + K * innovation;
       long int size = covariance.rows();
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(size, size);
        covariance = (I - K * C) * covariance;
    }
};

int main() {
    KalmanFilter filter;
    double deltaT = 0.1;
    double steeringAngleSpeed = 0.0;
    double velocityChange = 1.0;
    double wheel1Speed = 1.0;
    double wheel2Speed = 0.9;
    double wheel3Speed = 0.97;
    double wheel4Speed = 1.01;
    double steeringAngle = 0.98;

    
    
    
    filter.predict(deltaT, steeringAngleSpeed, velocityChange);
    filter.update(wheel1Speed, wheel2Speed, wheel3Speed, wheel4Speed, steeringAngle);

    std::cout << "Estimated position: (" << filter.state[0] << ", " << filter.state[1] << ")" << std::endl;
    std::cout << "Estimated heading: " << filter.state[2] * 180 / PI << " degrees" << std::endl;
    std::cout << "Estimated velocity: " << filter.state[3] << " m/s" << std::endl;

    return 0;
}

