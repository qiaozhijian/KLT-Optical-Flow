#include "parameters.h"

int MAX_CNT;
int MIN_DIST;
int FLOW_BACK;
int SHOW_FEATURE;
bool AFFINE = false;
bool WEIGHTED = false;
int MAX_LEVEL = 3;
bool USE_OPENCV = true;
double D_THRESHOLD = 0.5;

void readParameters(std::string config_file) {
    FILE *fh = fopen(config_file.c_str(), "r");
    if (fh == NULL) {
        std::cerr << "ERROR: config_file " << config_file << " does not exist." << std::endl;
        return;
    }
    fclose(fh);

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["affine"] >> AFFINE;
    fsSettings["weighted"] >> WEIGHTED;
    fsSettings["max_level"] >> MAX_LEVEL;

    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    D_THRESHOLD = fsSettings["D_threshold"];

    SHOW_FEATURE = fsSettings["show_feature"];
    FLOW_BACK = fsSettings["flow_back"];

    printf("Tracking parameters: \n");
    printf("USE_OPENCV: %d\n", USE_OPENCV);
    printf("MAX CNT: %d\n", MAX_CNT);
    printf("MIN DIST: %d\n", MIN_DIST);
    printf("AFFINE: %d\n", AFFINE);
    printf("WEIGHTED: %d\n", WEIGHTED);
    printf("MAX LEVEL: %d\n", MAX_LEVEL);

    printf("VO parameters: \n");
    printf("FLOW BACK: %d\n", FLOW_BACK);
    printf("D_THRESHOLD: %f\n", D_THRESHOLD);
    printf("SHOW FEATURE: %d\n", SHOW_FEATURE);

    fsSettings.release();
}
