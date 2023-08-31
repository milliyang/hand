#pragma once

#include <iterator>
#include <iostream>
#include <fstream>
#include <iomanip>  // to format image names using setw() and setfill()
#include <unistd.h>
#include <stdint.h>
#include <set>

#include "sort_def.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

#define USE_SSIM    (1)

class SimilarObj
{
public:
    SimilarObj();
    ~SimilarObj();

    void  init(cv::Mat &frame, const TrackingBox &tbox, int id);
    void  updateBox(const TrackingBox &tbox);
    void  update(cv::Mat &frame, const TrackingBox &tbox);
    float checkMatchScore(cv::Mat &frame, const RectBox &box);

public:
    int sort_id_;
    TrackingBox tbox_;

private:
    void generateSsimImage(cv::Mat &in, RectBox box, cv::Mat &out);
    void generateHogImage(cv::Mat &in, RectBox box, cv::Mat &out);

private:
    uint8_t     inited_;
    
    cv::Mat     mat_roi_;
};
