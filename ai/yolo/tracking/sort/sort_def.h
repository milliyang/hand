#pragma once

#include <iterator>
#include <iostream>
#include <fstream>
#include <iomanip>  // to format image names using setw() and setfill()
#include <unistd.h>
#include <stdint.h>
#include <set>
#include <string>

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using RectBox = cv::Rect_<float>;

typedef struct TrackingBox {
    int frame;
    int id;
    RectBox box;
    int class_idx;
    float confidence;
    std::string class_name;
} TrackingBox;

#define SORT_BOX_MIN            (8)
#define SORT_YOLO_SIZE          (416)
