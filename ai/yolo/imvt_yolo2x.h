#pragma once

#ifndef _IMVT_YOLO2X_H_
#define _IMVT_YOLO2X_H_

#include "yolo.h"

// Note:
// 1: For imvt_yolo2.h you need to check whether need to resize or region_box correction,
//    The Darknet Auther change format a lot.

// TODO:
//  both files need to be optimize for embedded platform, and avoid memory new/delete

detection *imvt_yolo2x_get_detection(struct region_layer *layer, int w, int h, float thresh, float hier, int *map, int relative, int *num,  int letter);
void imvt_yolo2x_free_detection(detection *dets, int n);
void imvt_yolo2x_do_nms_sort(detection *dets, int total, int classes, float thresh);

#endif