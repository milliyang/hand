#pragma once

typedef struct {
    int class_idx;
    char* class_name;
    float left;
    float top;
    float right;
    float bottom;
    float confidence;
} YoloBox;

typedef struct {
    float x, y, w, h;
} box;

typedef struct {
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

typedef enum {
    LAYER_DETECTION,
    LAYER_REGION,
    LAYER_YOLO,
    BLANK
} LAYER_TYPE;

struct yolov3_layer {
    int n;
    int w;
    int h;
    float *output;
    int outputs;
    int mask[3];
};

//to compatiable with yolo
struct region_layer {
    int type; //
    int n; //bbox_each_grid
    int w; //grid width
    int h; //grid height
    int batch;
    float biases[20]; //yolov2 10, yolov3 20
    int classes; //num of class
    int classfix;
    int coords;
    int background;
    float thresh;

    int net_w; //input_width
    int net_h; //input_height
    int re_entry;

    int outputs;
    float *output; //yolov2 output memory pointer

    int mask[3]; //012/345/678, 3 layer output for yolov3, should be equal n
    int sub_layer_num;
    int sub_layer_continue_memory;
    struct yolov3_layer sub_layer[3];
    //int outputs; // 13*13*
};

