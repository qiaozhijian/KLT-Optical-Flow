//
// Created by qzj on 23-4-7.
//

#ifndef CMAKE_TEMPLATE_BENCHMARK_H
#define CMAKE_TEMPLATE_BENCHMARK_H

#include <opencv2/opencv.hpp>

inline std::string padding(int num, int len) {
    std::string num_str = std::to_string(num);
    int num_len = num_str.length();
    for (int j = 0; j < len - num_len; ++j) {
        num_str = "0" + num_str;
    }
    return num_str;
}

inline void calcPrecisionRecall(const std::vector<cv::Point2f> &pts_est, const std::vector<cv::Point2f> &pts_opencv,
                         const std::vector<uchar> &status, const std::vector<uchar> &status_opencv, double &precision,
                         double &recall, double &f1) {
    assert(pts_est.size() == pts_opencv.size());
    int num_true_positives = 0;
    int num_positives = 0;
    int num_true = 0;
    for (int i = 0; i < status.size(); ++i) {
        if (status[i] == 1 && status_opencv[i] == 1) {
            if (cv::norm(pts_est[i] - pts_opencv[i]) < 2) {
                num_true_positives++;
            }
        }
        if (status[i] == 1) {
            num_positives++;
        }
        if (status_opencv[i] == 1) {
            num_true++;
        }
    }
    precision = num_true_positives / (double) num_positives;
    recall = num_true_positives / (double) num_true;
    if (precision + recall == 0)
        f1 = 0;
    else
        f1 = 2 * precision * recall / (precision + recall);
}

inline double distance(cv::Point2f pt1, cv::Point2f pt2) {
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

#endif //CMAKE_TEMPLATE_BENCHMARK_H
