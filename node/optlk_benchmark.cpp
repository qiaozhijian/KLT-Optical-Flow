#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "optFlowLK.h"
#include "parameters.h"
#include <chrono>
#include "benchmark_utils.h"
#include "global_defination/global_defination.h"

int main() {
    readParameters(cmake_template::WORK_SPACE_PATH+"/config/realsense_n3_unsync.yaml");
    std::string dir = cmake_template::WORK_SPACE_PATH+"/data/";
    std::string img1_path = dir + "frame" + padding(0, 4) + ".jpg";
    std::string img2_path = dir + "frame" + padding(10, 4) + ".jpg";
    cv::Mat img1 = cv::imread(img1_path);
    cv::Mat img2 = cv::imread(img2_path);
    // convert to gray image
    cv::Mat gray1, gray2;
    cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

    // detect features
    std::vector<cv::Point2f> keypoints1;
    int maxCorners = MAX_CNT;
    double qualityLevel = 0.01;
    double minDistance = MIN_DIST;
    cv::goodFeaturesToTrack(gray1, keypoints1, maxCorners, qualityLevel, minDistance);

    std::vector<cv::Point2f> keypoints2;
    std::vector<uchar> status;
    std::vector<float> err;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    elec5660::calcOpticalFlowPyrLK(gray1, gray2, keypoints1, keypoints2, status, err, cv::Size(WINDOW_SIZE, WINDOW_SIZE), MAX_LEVEL);
    if (FLOW_BACK) {
        std::vector<cv::Point2f> reversePts;
        std::vector<uchar> reverseStatus;
        elec5660::calcOpticalFlowPyrLK(gray2, gray1, keypoints2, reversePts, reverseStatus, err, cv::Size(WINDOW_SIZE, WINDOW_SIZE),
                                       MAX_LEVEL);
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i] && reverseStatus[i] && distance(reversePts[i], keypoints1[i]) <= D_THRESHOLD)
                status[i] = 1;
            else
                status[i] = 0;
        }
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    // OpenCV LK
    std::vector<cv::Point2f> keypoints2_opencv;
    std::vector<uchar> status_opencv;
    cv::calcOpticalFlowPyrLK(gray1, gray2, keypoints1, keypoints2_opencv, status_opencv, err, cv::Size(21, 21), 3);
    if (1) {
        std::vector<cv::Point2f> reversePts;
        std::vector<uchar> reverseStatus;
        cv::calcOpticalFlowPyrLK(gray2, gray1, keypoints2_opencv, reversePts, reverseStatus, err, cv::Size(21, 21), 3);
        for (size_t i = 0; i < status_opencv.size(); i++) {
            if (status_opencv[i] && reverseStatus[i] && distance(reversePts[i], keypoints1[i]) <= 0.5)
                status_opencv[i] = 1;
            else
                status_opencv[i] = 0;
        }
    }
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    double time = 0, time_opencv = 0;
    time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;
    time_opencv = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2).count() * 1000;

    double precision, recall, f1;
    calcPrecisionRecall(keypoints2, keypoints2_opencv, status, status_opencv, precision, recall, f1);
    std::cout << "frame id: " << 0 << ", precision: " << precision * 100 << "%, recall: " << recall * 100 << "%, f1: "
              << f1 * 100 << "%" << ", time: " << time << "ms, time_opencv: " << time_opencv << "ms" << std::endl;

    // draw result
    cv::Mat resultImg = cv::Mat::zeros(img1.rows, img1.cols * 2, CV_8UC3);
    cv::hconcat(img1, img2, resultImg);
    for (size_t i = 0; i < keypoints1.size(); i++) {
        if (status[i] == 0 || status_opencv[i] == 0)
            continue;
        cv::line(resultImg, keypoints1[i], keypoints2[i] + cv::Point2f(img1.cols, 0), cv::Scalar(255, 0, 0));
        cv::circle(resultImg, keypoints1[i], 2, cv::Scalar(0, 0, 255), 2);
        cv::circle(resultImg, keypoints2[i] + cv::Point2f(img1.cols, 0), 2, cv::Scalar(255, 0, 0), 2);
        cv::line(resultImg, keypoints1[i], keypoints2_opencv[i] + cv::Point2f(img1.cols, 0), cv::Scalar(0, 255, 0));
        cv::circle(resultImg, keypoints2_opencv[i] + cv::Point2f(img1.cols, 0), 2, cv::Scalar(0, 255, 255), 2);
    }

    // show result
    cv::imshow("KLT result", resultImg);
    cv::waitKey(0);
    std::string result_path = dir + "demo.jpg";
    cv::imwrite(result_path, resultImg);

    return 0;
}

