//
// Created by qzj on 23-3-31.
//

#ifndef SRC_OPTFLOWLK_H
#define SRC_OPTFLOWLK_H

#include "eigen_utils.h"
#include <opencv2/opencv.hpp>
#include "parameters.h"

namespace elec5660 {
    class OptFlowLK {
    private:
        cv::Size winSize;
        int maxLevel;
        cv::TermCriteria criteria;
        int flags;
        double minEigThreshold;
        std::vector<cv::Mat> prevGradXPyr, prevGradYPyr, nextGradXPyr, nextGradYPyr;
    public:
        OptFlowLK() = default;

        OptFlowLK(cv::Size winSize = cv::Size(21, 21), int pyrLevel = 3,
                  cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                  int flags = 0, double minEigThreshold = 1e-4) :winSize(winSize), criteria(criteria), flags(flags), minEigThreshold(minEigThreshold) {
            maxLevel = pyrLevel;
        }

        void calc(const cv::Mat& prevImg, const cv::Mat& nextImg,
                  const std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& nextPts,
                  std::vector<uchar>& status, std::vector<float>& err){
            // 构建金字塔
            std::vector<cv::Mat> prevPyramid, nextPyramid;
            buildPyramid(prevImg, prevPyramid, prevGradXPyr, prevGradYPyr, maxLevel);
            buildPyramid(nextImg, nextPyramid, nextGradXPyr, nextGradYPyr, maxLevel);
            // 计算光流并绘制跟踪结果
            int windowSize = winSize.width;
            nextPts.clear();
            for (size_t i = 0; i < prevPts.size(); i++) {
                cv::Point2f displacement = calcOpticalFlow(prevPyramid, nextPyramid, prevPts[i], windowSize, maxLevel);
                cv::Point2f keypoints2i = prevPts[i] + displacement;
                nextPts.push_back(keypoints2i);
                // compute error
                if (!inBorder(cv::Point2f(prevPts[i].x - windowSize / 2, prevPts[i].y - windowSize / 2), prevImg.size()) ||
                    !inBorder(cv::Point2f(prevPts[i].x + windowSize / 2, prevPts[i].y + windowSize / 2), prevImg.size()) ||
                    !inBorder(cv::Point2f(keypoints2i.x - windowSize / 2, keypoints2i.y - windowSize / 2), nextImg.size()) ||
                    !inBorder(cv::Point2f(keypoints2i.x + windowSize / 2, keypoints2i.y + windowSize / 2), nextImg.size())){
                    err.push_back(1e10);
                } else{
                    cv::Mat patch = prevImg(cv::Rect(prevPts[i].x - windowSize / 2, prevPts[i].y - windowSize / 2, windowSize, windowSize));
                    cv::Mat patchNext = nextImg(cv::Rect(keypoints2i.x - windowSize / 2, keypoints2i.y - windowSize / 2, windowSize, windowSize));
                    cv::Mat patchDiff = patch - patchNext;
                    err.push_back(cv::sqrt(cv::mean(patchDiff.mul(patchDiff))[0]));
                }
                if (err[i] < 10.0) {
                    status.push_back(1);
                } else {
                    status.push_back(0);
                }
            }
        }

        double calcPatchDiff(const cv::Mat& prevImg, const cv::Mat& nextImg,
                             const cv::Point2f& prevPt, const cv::Point2f& nextPt){

            // 计算光流并绘制跟踪结果
            int windowSize = winSize.width;
            if (!inBorder(cv::Point2f(prevPt.x - windowSize / 2, prevPt.y - windowSize / 2), prevImg.size()) ||
                !inBorder(cv::Point2f(prevPt.x + windowSize / 2, prevPt.y + windowSize / 2), prevImg.size()) ||
                !inBorder(cv::Point2f(nextPt.x - windowSize / 2, nextPt.y - windowSize / 2), nextImg.size()) ||
                !inBorder(cv::Point2f(nextPt.x + windowSize / 2, nextPt.y + windowSize / 2), nextImg.size())){
                return 1e10;
            }
            cv::Mat patch = cv::Mat::zeros(windowSize, windowSize, CV_32F);
            for (int v = 0; v < windowSize; v++) {
                for (int u = 0; u < windowSize; u++) {
                    patch.at<float>(v, u) = cvMatAt(prevImg, prevPt.x - windowSize / 2 + u, prevPt.y - windowSize / 2 + v);
                }
            }
            cv::Mat patchNext = cv::Mat::zeros(windowSize, windowSize, CV_32F);
            for (int v = 0; v < windowSize; v++) {
                for (int u = 0; u < windowSize; u++) {
                    patchNext.at<float>(v, u) = cvMatAt(nextImg, nextPt.x - windowSize / 2 + u, nextPt.y - windowSize / 2 + v);
                }
            }
            //cv::Mat patch = prevImg(cv::Rect(prevPt.x - windowSize / 2, prevPt.y - windowSize / 2, windowSize, windowSize));
            //cv::Mat patchNext = nextImg(cv::Rect(nextPt.x - windowSize / 2, nextPt.y - windowSize / 2, windowSize, windowSize));
            cv::Mat patchDiff = patch - patchNext;
            double err = cv::sqrt(cv::mean(patchDiff.mul(patchDiff))[0]);
            return err;
        }

        void buildPyramid(const cv::Mat &src, std::vector<cv::Mat> &pyramid, std::vector<cv::Mat> &gradXPyr, std::vector<cv::Mat> &gradYPyr, int levels) {
            pyramid.resize(levels);
            gradXPyr.resize(levels);
            gradYPyr.resize(levels);
            pyramid[0] = src.clone();
            cv::Mat scharr_kernel_x = (cv::Mat_<float>(3, 3) << -3, 0, 3, -10, 0, 10, -3, 0, 3) / 16.0 / 2.0;
            cv::Mat scharr_kernel_y = (cv::Mat_<float>(3, 3) << -3, -10, -3, 0, 0, 0, 3, 10, 3) / 16.0 / 2.0;
            cv::filter2D(pyramid[0], gradXPyr[0], CV_16SC1, scharr_kernel_x);
            cv::filter2D(pyramid[0], gradYPyr[0], CV_16SC1, scharr_kernel_y);
            for (int i = 1; i < levels; ++i) {
                cv::pyrDown(pyramid[i - 1], pyramid[i]);
                cv::filter2D(pyramid[i], gradXPyr[i], CV_16SC1, scharr_kernel_x);
                cv::filter2D(pyramid[i], gradYPyr[i], CV_16SC1, scharr_kernel_y);
            }
            for (int i = 0; i < levels; ++i) {
                pyramid[i].convertTo(pyramid[i], CV_32FC1);
                gradXPyr[i].convertTo(gradXPyr[i], CV_32FC1);
                gradYPyr[i].convertTo(gradYPyr[i], CV_32FC1);
            }
            //    test scharr
            //    int u = 100, v = 100;
            //    cv::Mat patch = pyramid[0](cv::Rect(u - 1, v - 1, 3, 3));
            //    patch.convertTo(patch, CV_32FC1);
            //    double sum = cv::sum(patch.mul(scharr_kernel_x))[0] / 16.0 / 2.0;
            //    std::cout << "sum: " << sum << " gradX: " << gradXPyr[0].at<float>(v, u) << std::endl;
            //    while (1);
        }

        bool inBorder(const cv::Point2f &pt, const cv::Size &size) {
            return pt.x >= 0 && pt.x < size.width && pt.y >= 0 && pt.y < size.height;
        }

        cv::Point2f calcOpticalFlowSingleLevel(const cv::Mat &prev, const cv::Mat &next,
                                               const cv::Point2f &point, const cv::Point2f &pointPred,
                                               int windowSize, int level) {
            float halfWindow = (float)windowSize / 2.f;
            if (!inBorder(cv::Point2f(point.x - halfWindow, point.y - halfWindow), prev.size()) ||
                !inBorder(cv::Point2f(point.x + halfWindow, point.y + halfWindow), prev.size()) ||
                !inBorder(cv::Point2f(point.x - halfWindow, point.y - halfWindow), next.size()) ||
                !inBorder(cv::Point2f(point.x + halfWindow, point.y + halfWindow), next.size())){
                return cv::Point2f(0.f, 0.f);
            }
            cv::Mat patch = prev(cv::Rect(int(point.x - halfWindow), int(point.y - halfWindow), windowSize, windowSize));
            cv::Mat patchNext = next(cv::Rect(int(pointPred.x - halfWindow), int(pointPred.y - halfWindow), windowSize, windowSize));
            cv::Mat patchGradX = prevGradXPyr[level](cv::Rect(int(point.x - halfWindow), int(point.y - halfWindow), windowSize, windowSize));
            cv::Mat patchGradY = prevGradYPyr[level](cv::Rect(int(point.x - halfWindow), int(point.y - halfWindow), windowSize, windowSize));
            cv::Mat patchGradXNext = nextGradXPyr[level](cv::Rect(int(pointPred.x - halfWindow), int(pointPred.y - halfWindow), windowSize, windowSize));
            cv::Mat patchGradYNext = nextGradYPyr[level](cv::Rect(int(pointPred.x - halfWindow), int(pointPred.y - halfWindow), windowSize, windowSize));
            Eigen::Matrix2d G = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();

            double weight_sum = 0;
            for (int y = 0; y < patch.rows; ++y) {
                for (int x = 0; x < patch.cols; ++x) {
                    Eigen::Vector2d JNext(cvMatAt(patchGradXNext, x, y), cvMatAt(patchGradYNext, x, y));
                    double weight = 1.0;
                    if (WEIGHTED){
                        Eigen::Vector2d J(patchGradX.at<float>(y, x), patchGradY.at<float>(y, x));
                        weight = 1.0 / ((J - JNext).norm() + 1.0);
                    }

                    G += JNext * JNext.transpose() * weight;
                    double di = cvMatAt(patch, x, y) - cvMatAt(patchNext, x, y);
                    b = b + di * JNext * weight;
                    weight_sum += weight;
                }
            }
            G = G / weight_sum;
            b = b / weight_sum;
            Eigen::Vector2d dp = G.inverse() * b;
            return cv::Point2f(dp.x(), dp.y());
        }

        cv::Point2f calcOpticalFlow(const std::vector<cv::Mat> &prevPyramid, const std::vector<cv::Mat> &nextPyramid, const cv::Point2f &point, int windowSize, int levels) {
            cv::Point2f displacement(0, 0);
            for (int level = levels - 1; level >= 0; --level) {
                cv::Mat prevLevel = prevPyramid[level];
                cv::Mat nextLevel = nextPyramid[level];

                cv::Point2f pointOnLevel = point * (1.0 / (1 << level));
                cv::Point2f pointOnLevelPred = (point + displacement) * (1.0 / (1 << level));
                double error_org = calcPatchDiff(prevPyramid[level], nextPyramid[level], pointOnLevel, pointOnLevel);
                for (int iter = 0; iter < 10; ++iter) {
                    cv::Point2f flowOnLevel;
                    flowOnLevel = calcOpticalFlowSingleLevel(prevLevel, nextLevel, pointOnLevel, pointOnLevelPred, windowSize, level);
                    pointOnLevelPred = pointOnLevelPred + flowOnLevel;
                    double updated_error = calcPatchDiff(prevPyramid[level], nextPyramid[level], pointOnLevel, pointOnLevelPred);
                    if (updated_error > error_org + 1e-3) {
                        pointOnLevelPred = pointOnLevelPred - flowOnLevel;
                        break;
                    }
                    error_org = updated_error;
                }
                displacement = (pointOnLevelPred - pointOnLevel) * (1 << level);
            }
            return displacement;
        }

        float cvMatAt(const cv::Mat &mat, float u, float v) {
            //return mat.at<float>(v, u);
            //    bilinear interpolation
            int x = floor(u);
            int y = floor(v);
            float dx = u - x;
            float dy = v - y;
            float value = mat.at<float>(y, x);
            if (x >= 0 && x < mat.cols - 1 && y >= 0 && y < mat.rows - 1) {
                value = (1 - dx) * (1 - dy) * mat.at<float>(y, x) +
                        dx * (1 - dy) * mat.at<float>(y, x + 1) +
                        (1 - dx) * dy * mat.at<float>(y + 1, x) +
                        dx * dy * mat.at<float>(y + 1, x + 1);
            } else {
                value = 0;
            }
            return value;
        }
    };

    inline void calcOpticalFlowPyrLK(const cv::Mat& prevImg, const cv::Mat& nextImg,
                                     const std::vector<cv::Point2f>& prevPts, std::vector<cv::Point2f>& nextPts,
                                     std::vector<uchar>& status, std::vector<float>& err,
                                     cv::Size winSize = cv::Size(21, 21), int maxLevel = 3,
                                     cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                                     int flags = 0, double minEigThreshold = 1e-4){
        OptFlowLK optFlowLK(winSize, MAX_LEVEL, criteria, flags, minEigThreshold);
        optFlowLK.calc(prevImg, nextImg, prevPts, nextPts, status, err);
    }
}

#endif //SRC_OPTFLOWLK_H
