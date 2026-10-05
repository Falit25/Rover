#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace lunar_slam {

Eigen::Matrix3d hat(const Eigen::Vector3d& w);
Eigen::Matrix3d exp_so3(const Eigen::Vector3d& w);

Eigen::Quaterniond updateQuaternion(const Eigen::Quaterniond& q, const Eigen::Vector3d& delta_omega);


Eigen::Matrix4d createSE3Matrix(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

}