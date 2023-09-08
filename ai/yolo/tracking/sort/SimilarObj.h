#pragma once

#include "sort_def.h"

#define METHOD_SSIM         (0)
#define METHOD_HOG          (1)
#define METHOD_SIFT         (2)

#define MATCH_METHOD        (2)

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
    void generateSsimImage(cv::Mat &in, const RectBox &box, cv::Mat &out);
    void generateHogImage(cv::Mat &in,  const RectBox &box, cv::Mat &out);
    void generateCropImage(cv::Mat &in, const RectBox &box, cv::Mat &out);

private:
    uint8_t     inited_;
    cv::Mat     mat_roi_;
};
