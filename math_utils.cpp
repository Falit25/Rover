#include "lunar_slam/math_utils.hpp"
#include <cmath>

namespace lunar_slam {

Eigen::Matrix3d hat(const Eigen::Vector3d& w) {
    Eigen::Matrix3d w_hat;
    w_hat <<     0.0, -w.z(),  w.y(),
               w.z(),    0.0, -w.x(),
              -w.y(),  w.x(),    0.0;
    return w_hat;
}

Eigen::Matrix3d exp_so3(const Eigen::Vector3d& w) {
    double theta = w.norm();
    
    if (theta < 1e-7) {
        return Eigen::Matrix3d::Identity() + hat(w);
    }

    Eigen::Vector3d u = w / theta;
    Eigen::Matrix3d u_hat = hat(u);

    return Eigen::Matrix3d::Identity() + 
           std::sin(theta) * u_hat + 
           (1.0 - std::cos(theta)) * (u_hat * u_hat);
}

Eigen::Quaterniond updateQuaternion(const Eigen::Quaterniond& q, const Eigen::Vector3d& delta_omega) {
    double delta_theta = delta_omega.norm();
    if (delta_theta < 1e-7) {
        return q.normalized();
    }
    
    Eigen::Vector3d delta_axis = delta_omega.normalized();
    Eigen::Quaterniond delta_q(Eigen::AngleAxisd(delta_theta, delta_axis));

    Eigen::Quaterniond q_updated = q * delta_q;
    return q_updated.normalized(); 
}

Eigen::Matrix4d createSE3Matrix(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = R; 
    T.block<3,1>(0,3) = t; 
    return T;
}

}