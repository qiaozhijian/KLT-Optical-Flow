//
// Created by qzj on 23-3-24.
//

#ifndef SRC_EIGEN_UTILS_H
#define SRC_EIGEN_UTILS_H

#include <Eigen/Core>
#include <random>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

inline double random_normal(double std) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0, std);
    double number = dis(gen);
    return number;
}

inline Eigen::Vector3d random_unit_vector() {
    Eigen::Vector3d v;
    v << random_normal(1.0), random_normal(1.0), random_normal(1.0);
    v.normalize();
    return v;
}

inline Eigen::Quaterniond random_unit_quaternion() {
    Eigen::Quaterniond q;
    q.w() = random_normal(1);
    q.vec() = random_unit_vector();
    q.normalize();
    return q;
}

inline Eigen::Matrix3d random_rotation_matrix() {
    Eigen::Quaterniond q = random_unit_quaternion();
    Eigen::Matrix3d R = q.toRotationMatrix();
    return R;
}

inline Eigen::Matrix<double, 3, 3> cv2eigen(const cv::Mat &cv_mat) {
    Eigen::Matrix<double, 3, 3> eigen_mat = Eigen::Matrix<double, 3, 3>::Identity();
    eigen_mat << cv_mat.at<double>(0, 0), cv_mat.at<double>(0, 1), cv_mat.at<double>(0, 2),
            cv_mat.at<double>(1, 0), cv_mat.at<double>(1, 1), cv_mat.at<double>(1, 2),
            cv_mat.at<double>(2, 0), cv_mat.at<double>(2, 1), cv_mat.at<double>(2, 2);
    return eigen_mat;
}

template<typename T>
double RotationDistance(Eigen::Matrix<T, 3, 3> &R1, Eigen::Matrix<T, 3, 3> &R2) {
    Eigen::Matrix<T, 3, 3> R = R1 * R2.transpose();
    Eigen::AngleAxis<T> angle_axis(R);
    return angle_axis.angle() * 180 / M_PI;
}

template<typename T>
double RotationDistance(Eigen::Matrix<T, 4, 4> &T1, Eigen::Matrix<T, 4, 4> &T2) {
    Eigen::Matrix<T, 3, 3> R1 = T1.block(0, 0, 3, 3);
    Eigen::Matrix<T, 3, 3> R2 = T2.block(0, 0, 3, 3);
    Eigen::Matrix<T, 3, 3> R = R1 * R2.transpose();
    Eigen::AngleAxis<T> angle_axis(R);
    return angle_axis.angle() * 180 / M_PI;
}

template<typename T>
Eigen::Matrix<T, 3, 3> skew(Eigen::Matrix<T, 3, 1> &v) {
    Eigen::Matrix<T, 3, 3> skew;
    skew << 0, -v(2), v(1),
            v(2), 0, -v(0),
            -v(1), v(0), 0;
    return skew;
}

template<typename T>
int TwoSigmaRule(const std::vector<T> &errors, std::vector<int> &inliers_idx){
    T mean = 0;
    T sigma = 0;
    for (int i = 0; i < errors.size(); ++i) {
        mean += errors[i];
    }
    mean /= errors.size();
    for (int i = 0; i < errors.size(); ++i) {
        sigma += (errors[i] - mean) * (errors[i] - mean);
    }
    sigma = sqrt(sigma / errors.size());
    std::cout << "mean: " << mean << " sigma: " << sigma << std::endl;
    int inlier_num = 0;
    for (int i = 0; i < errors.size(); ++i) {
        if (errors[i] < mean + 2 * sigma && errors[i] > mean - 2 * sigma) {
            inliers_idx.push_back(i);
            inlier_num++;
        }
    }
    return inlier_num;
}

#endif //SRC_EIGEN_UTILS_H
