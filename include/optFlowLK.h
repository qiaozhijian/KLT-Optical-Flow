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
                    !inBorder(cv::Point2f(prevPts[i].x + windowSize / 2, prevPts[i].y + windowSize / 2), prevImg.size())){
                    err.push_back(1e10);
                } else{
                    cv::Mat patch = prevImg(cv::Rect(prevPts[i].x - windowSize / 2, prevPts[i].y - windowSize / 2, windowSize, windowSize));
                    cv::Mat patchNext = nextImg(cv::Rect(keypoints2i.x - windowSize / 2, keypoints2i.y - windowSize / 2, windowSize, windowSize));
                    cv::Mat patchDiff = patch - patchNext;
                    err.push_back(cv::sqrt(cv::mean(patchDiff.mul(patchDiff))[0]));
                }
                if (err[i] < 10.0 && inBorder(keypoints2i, nextImg.size())) {
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
            cv::Mat patch = prevImg(cv::Rect(prevPt.x - windowSize / 2, prevPt.y - windowSize / 2, windowSize, windowSize));
            cv::Mat patchNext = nextImg(cv::Rect(nextPt.x - windowSize / 2, nextPt.y - windowSize / 2, windowSize, windowSize));
            cv::Mat patchDiff = patch - patchNext;
            double err = cv::sqrt(cv::mean(patchDiff.mul(patchDiff))[0]);
            return err;
        }

        void buildPyramid(const cv::Mat &src, std::vector<cv::Mat> &pyramid, std::vector<cv::Mat> &gradXPyr, std::vector<cv::Mat> &gradYPyr, int levels) {
            pyramid.resize(levels);
            gradXPyr.resize(levels);
            gradYPyr.resize(levels);
            pyramid[0] = src.clone();
            cv::Sobel(pyramid[0], gradXPyr[0], CV_32F, 1, 0, 3);
            cv::Sobel(pyramid[0], gradYPyr[0], CV_32F, 0, 1, 3);
            for (int i = 1; i < levels; ++i) {
                cv::pyrDown(pyramid[i - 1], pyramid[i]);
                cv::Sobel(pyramid[i], gradXPyr[i], CV_32F, 1, 0, 3);
                cv::Sobel(pyramid[i], gradYPyr[i], CV_32F, 0, 1, 3);
            }
            for (int i = 0; i < levels; ++i) {
                pyramid[i].convertTo(pyramid[i], CV_32F);
                gradXPyr[i].convertTo(gradXPyr[i], CV_32F);
                gradYPyr[i].convertTo(gradYPyr[i], CV_32F);
            }
        }

        bool inBorder(const cv::Point2f &pt, const cv::Size &size) {
            return pt.x >= 0 && pt.x < size.width && pt.y >= 0 && pt.y < size.height;
        }

        cv::Point2f calcOpticalFlowSingleLevel(const cv::Mat &prev, const cv::Mat &next, const cv::Point2f &point, int windowSize, int level) {
            int halfWindow = windowSize / 2;
            if (!inBorder(cv::Point2f(point.x - halfWindow, point.y - halfWindow), prev.size()) ||
                !inBorder(cv::Point2f(point.x + halfWindow, point.y + halfWindow), prev.size())) {
                return cv::Point2f(0, 0);
            }
            cv::Mat patch = prev(cv::Rect(point.x - halfWindow, point.y - halfWindow, windowSize, windowSize));
            cv::Mat patchNext = next(cv::Rect(point.x - halfWindow, point.y - halfWindow, windowSize, windowSize));
            cv::Mat patchGradX, patchGradY, patchGradXNext, patchGradYNext;
            patchGradX = prevGradXPyr[level](cv::Rect(point.x - halfWindow, point.y - halfWindow, windowSize, windowSize));
            patchGradY = prevGradYPyr[level](cv::Rect(point.x - halfWindow, point.y - halfWindow, windowSize, windowSize));
            patchGradXNext = nextGradXPyr[level](cv::Rect(point.x - halfWindow, point.y - halfWindow, windowSize, windowSize));
            patchGradYNext = nextGradYPyr[level](cv::Rect(point.x - halfWindow, point.y - halfWindow, windowSize, windowSize));
            //cv::Sobel(patch, patchGradX, CV_32F, 1, 0, 3);
            //cv::Sobel(patch, patchGradY, CV_32F, 0, 1, 3);
            //cv::Sobel(patchNext, patchGradXNext, CV_32F, 1, 0, 3);
            //cv::Sobel(patchNext, patchGradYNext, CV_32F, 0, 1, 3);
            Eigen::Matrix2d G = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();

            double weight_sum = 0;
            for (int y = 0; y < patch.rows; ++y) {
                for (int x = 0; x < patch.cols; ++x) {
                    float dx = patchGradX.at<float>(y, x);
                    float dy = patchGradY.at<float>(y, x);
                    Eigen::Vector2d J(dx, dy);
                    Eigen::Vector2d JNext(patchGradXNext.at<float>(y, x), patchGradYNext.at<float>(y, x));
                    double weight = 1.0;
                    if (WEIGHTED)
                        weight = 1.0 / ((J - JNext).norm() + 1.0);
                    G += J * J.transpose() * weight;

                    double di = patchNext.at<float>(y, x) - patch.at<float>(y, x);
                    b = b - di * J * weight;
                    weight_sum += weight;
                }
            }
            G = G / weight_sum;
            b = b / weight_sum;
            Eigen::Vector2d dp = G.inverse() * b;
            return cv::Point2f(dp.x(), dp.y());
        }


        cv::Point2f calcOpticalFlowSingleLevelAffine(const cv::Mat &prev, const cv::Mat &next, const cv::Point2f &point, int windowSize, int level) {
            int halfWindow = windowSize / 2;
            if (!inBorder(cv::Point2f(point.x - halfWindow, point.y - halfWindow), prev.size()) ||
                !inBorder(cv::Point2f(point.x + halfWindow, point.y + halfWindow), prev.size())) {
                return cv::Point2f(0, 0);
            }
            cv::Mat patch = prev(cv::Rect(point.x - halfWindow, point.y - halfWindow, windowSize, windowSize));
            cv::Mat patchNext = next(cv::Rect(point.x - halfWindow, point.y - halfWindow, windowSize, windowSize));
            cv::Mat patchGradX, patchGradY, patchGradXNext, patchGradYNext;
            patchGradX = prevGradXPyr[level](cv::Rect(point.x - halfWindow, point.y - halfWindow, windowSize, windowSize));
            patchGradY = prevGradYPyr[level](cv::Rect(point.x - halfWindow, point.y - halfWindow, windowSize, windowSize));
            patchGradXNext = nextGradXPyr[level](cv::Rect(point.x - halfWindow, point.y - halfWindow, windowSize, windowSize));
            patchGradYNext = nextGradYPyr[level](cv::Rect(point.x - halfWindow, point.y - halfWindow, windowSize, windowSize));
            //cv::Sobel(patch, patchGradX, CV_32F, 1, 0, 3);
            //cv::Sobel(patch, patchGradY, CV_32F, 0, 1, 3);
            //cv::Sobel(patchNext, patchGradXNext, CV_32F, 1, 0, 3);
            //cv::Sobel(patchNext, patchGradYNext, CV_32F, 0, 1, 3);
            Eigen::Matrix<double, 6, 6> G = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

            double weight_sum = 0;
            for (int y = 0; y < patch.rows; ++y) {
                for (int x = 0; x < patch.cols; ++x) {
                    float dx = patchGradX.at<float>(y, x);
                    float dy = patchGradY.at<float>(y, x);
                    Eigen::Matrix<double, 1, 6> J;
                    J << x * dx, y * dx, dx, x * dy, y * dy, dy;

                    Eigen::Vector2d grad(patchGradX.at<float>(y, x), patchGradY.at<float>(y, x));
                    Eigen::Vector2d gradNext(patchGradXNext.at<float>(y, x), patchGradYNext.at<float>(y, x));
                    double weight = 1.0;
                    if (WEIGHTED)
                        weight = 1.0 / ((grad - gradNext).norm() + 1.0);
                    G += J.transpose() * J * weight;

                    double di = patchNext.at<float>(y, x) - patch.at<float>(y, x);
                    b = b - di * J.transpose() * weight;
                    weight_sum += weight;
                }
            }
            G = G / weight_sum;
            b = b / weight_sum;
            Eigen::Matrix<double, 6, 1> dp = G.inverse() * b;
            Eigen::Matrix2d A;
            A << 1 + dp(0), dp(1), dp(2), 1 + dp(3);
            Eigen::Vector2d b2(dp(4), dp(5));
            Eigen::Vector2d displacement = A * Eigen::Vector2d(point.x, point.y) + b2 - Eigen::Vector2d(point.x, point.y);
            return cv::Point2f(displacement.x(), displacement.y());
        }

        cv::Point2f calcOpticalFlow(const std::vector<cv::Mat> &prevPyramid, const std::vector<cv::Mat> &nextPyramid, const cv::Point2f &point, int windowSize, int levels) {
            //std::cout << "levels: " << levels << std::endl;
            // compute initial patch difference on every level
            for (int level = levels - 1; level >= 0; --level) {
                cv::Mat prevLevel = prevPyramid[level];
                cv::Mat nextLevel = nextPyramid[level];
                cv::Point2f pointOnLevel = point * (1.0 / (1 << level));
                //std::cout << "prev error: " << calcPatchDiff(prevPyramid[level], nextPyramid[level], pointOnLevel, pointOnLevel) << ", level: " << level << std::endl;
            }

            cv::Point2f displacement(0, 0);
            for (int level = levels - 1; level >= 0; --level) {
                cv::Point2f pointOnLevel = (point + displacement) * (1.0 / (1 << level));
                cv::Mat prevLevel = prevPyramid[level];
                cv::Mat nextLevel = nextPyramid[level];

                cv::Point2f displacementOnLevel(0, 0);
                cv::Point2f displacementOnLevelTmp(0, 0);
                cv::Point2f pointOnLevelOrig = point * (1.0 / (1 << level));
                double last_error = calcPatchDiff(prevPyramid[level], nextPyramid[level], pointOnLevelOrig, pointOnLevel);
                for (int iter = 0; iter < 10; ++iter) {
                    cv::Point2f pointOnLevelInNext = pointOnLevel + displacementOnLevel;
                    cv::Point2f flowOnLevel;
                    if (AFFINE)
                        flowOnLevel = calcOpticalFlowSingleLevelAffine(prevLevel, nextLevel, pointOnLevelInNext, windowSize, level);
                    else
                        flowOnLevel = calcOpticalFlowSingleLevel(prevLevel, nextLevel, pointOnLevelInNext, windowSize, level);
                    double updated_error = calcPatchDiff(prevPyramid[level], nextPyramid[level], pointOnLevelOrig, pointOnLevelInNext + flowOnLevel);
                    //std::cout << "iter: " << iter << " level: " << level << " last_error: " << last_error << " updated_error: " << updated_error << std::endl;
                    if (updated_error > last_error + 1e-3) {
                        displacementOnLevel -= displacementOnLevelTmp;
                        break;
                    } else if (updated_error < last_error - 1e-3) {
                        displacementOnLevelTmp = cv::Point2f(0, 0);
                    } else {
                        displacementOnLevelTmp += flowOnLevel;
                    }
                    last_error = updated_error;
                    displacementOnLevel += flowOnLevel;
                }
                //std::cout << "level: " << level << " displacement: " << displacementOnLevel << std::endl;
                displacement += displacementOnLevel * (1 << level);
            }
            //std::cout << "final error: " << calcPatchDiff(prevPyramid[0], nextPyramid[0], point, point + displacement) << std::endl;
            //std::cout << "final displacement: " << displacement << std::endl;
            return displacement;
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
