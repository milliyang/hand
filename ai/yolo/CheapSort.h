#pragma once

#include <iterator>
#include <iostream>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()
//#include <io.h>    // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#include <unistd.h>
#include <stdint.h>
#include <set>

#include "Hungarian.h"
#include "KalmanTracker.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

#define TRK_NUM                 20
#define TRK_RESET_ID_FRAME_NUM  100

#define TRK_EXPIRE_MAX_AGE          (100)
#define TRK_IOU_THRESHOLD           (0.3f)

typedef struct TrackingBox {
    int frame;
    int id;
    cv::Rect_<float> box;

    int class_idx;
    string class_name;
    float confidence;
} TrackingBox;

class CheapSort
{
public:
    CheapSort();
    ~CheapSort();

    std::vector<TrackingBox> Run(std::vector<TrackingBox> t_boxes);
    void Clear();

    static void KalmanGlobalResetId(void);

private:
    void Init(std::vector<TrackingBox> t_boxes);

    // update across frames
    int frame_count_;
    int max_age_;
    int min_hits_;
    float iou_threshold_;
    std::vector<KalmanTracker> trackers_;

    // global variables for counting
    float total_time_;
    std::vector<TrackingBox> tracking_result_;

    int no_tracker_count_;  //
};
