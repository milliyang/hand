#pragma once

#ifndef BOX_H
#define BOX_H

#include "yolo.h"

typedef struct {
    float dx, dy, dw, dh;
} dbox;

float box_rmse(box a, box b);
dbox diou(box a, box b);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);
float box_iou(box a, box b);

void do_nms_sort(detection *dets, int total, int classes, float thresh);

#endif
