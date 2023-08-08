#pragma once

#include <iterator>
#include <iostream>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()
//#include <io.h>    // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#include <unistd.h>
#include <set>

#include "Hungarian.h"
#include "KalmanTracker.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

#define CNUM 20

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

    vector<TrackingBox> Run(vector<TrackingBox> t_boxes);
    void Clear();

    static void KalmanGlobalReset(void);

private:
    // 3. update across frames
    int frame_count;
    int max_age;
    int min_hits;
    double iouThreshold;
    vector<KalmanTracker> trackers;

    void Init(vector<TrackingBox> t_boxes);

    // global variables for counting
    int total_frames;
    double total_time;

    vector<TrackingBox> tracking_result;
};

