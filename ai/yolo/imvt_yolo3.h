#ifndef _IMVT_YOLO3_H_
#define _IMVT_YOLO3_H_

#include "yolo.h"

#define IMVT_YOLO_QUANT_BASE_CAFFE (0)
#define IMVT_YOLO_QUANT_BASE_HISI  (4096)

int imvt_yolo3_init(struct region_layer *layer, void* mem_buffer, int size, int quant_base);
int imvt_yolo3_deinit(void);

detection *imvt_yolo3_get_detection(struct region_layer *layer, int w, int h, float thresh, float hier, int *map, int relative, int *num);
void imvt_yolo3_free_detection(detection *dets, int n); //deprecated, use init() & deinit()
void imvt_yolo3_do_nms_sort(detection *dets, int total, int classes, float thresh);

#endif
