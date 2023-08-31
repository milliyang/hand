#include "SimilarTracker.h"

#ifdef CONFIG_SPDLOG
#define LOG_TAG "SimTrker"
#include "log.h"
#endif

#define SSIM_FEATURE_SIZE   (8)

static bool check_box_valid(const cv::Rect_<float> &bb)
{
    if (bb.x >= 0 && bb.y >= 0) {
        return true;
    } else {
        //bb.x == nan
        return false;
    }
}

// Computes IOU between two bounding boxes
static float GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;
    if (un < DBL_EPSILON) {
        return 0;
    }
    return (float)(in / un);
}

SimilarTracker::SimilarTracker(void)
{

}

SimilarTracker::~SimilarTracker(void)
{

}

TrackingBox SimilarTracker::Run(cv::Mat &frame, std::vector<TrackingBox> tboxes)
{

}

/*start tracking by ID/Rect */
void trackById(int id)
{

}

void trackByRect(const RectBox &rect)
{

}
