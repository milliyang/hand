#pragma once

#include <iterator>
#include <iostream>
#include <fstream>
#include <iomanip>  // to format image names using setw() and setfill()
#include <unistd.h>
#include <stdint.h>
#include <set>
#include <string>
#include <map>

#include "Hungarian.h"
#include "KalmanTracker.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using RectBox = cv::Rect_<float>;
using IntMap  = std::map<int,int>;

typedef struct TrackingBox {
    int      frame;
    uint8_t  tracking;
    uint8_t  reserved;
    uint16_t class_idx;
    int      id;
    RectBox     box;
    float       confidence;
    std::string class_name;
} TrackingBox;

#define SORT_BOX_MIN                (8)
#define SORT_YOLO_SIZE              (416)
#define SORT_YOLO_CLASS_NUM         (20)

#define SORT_OBJECT_AGE_MAX             (90)        //3s
#define SORT_OBJECT_HIDDEN_AGE_MAX      (6)         //5frame

#define SORT_CLS_HUMAN (0)
#define SORT_CLS_FACE  (1)
#define SORT_CLS_HAND  (2)