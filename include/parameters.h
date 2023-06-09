#pragma once

#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

extern int MAX_CNT;
extern int MIN_DIST;
extern int SHOW_FEATURE;
extern int FLOW_BACK;
extern bool AFFINE;
extern bool WEIGHTED;
extern int MAX_LEVEL;
extern bool USE_OPENCV;
extern double D_THRESHOLD;
extern int WINDOW_SIZE;

void readParameters(std::string config_file);
