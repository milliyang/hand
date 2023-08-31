#pragma once

#include <iterator>
#include <iostream>
#include <fstream>
#include <iomanip>  // to format image names using setw() and setfill()
#include <unistd.h>
#include <stdint.h>
#include <set>

#include "sort_def.h"
#include "SimilarObj.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

class SimilarTracker
{
public:
    SimilarTracker();
    ~SimilarTracker();

    /*start tracking by ID/Rect */
    void trackById(int id);
    void trackByRect(const RectBox &rect);

    TrackingBox Run(cv::Mat &frame, std::vector<TrackingBox> tboxes);
    //TrackingBox Run(std::vector<TrackingBox> tboxes);
    TrackingBox Get(void);  /*get laster result*/

private:

private:
    uint32_t uuid_;
    std::vector<SimilarObj> objects_;
};
