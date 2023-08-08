//#include "stdafx.h"
#include "imvt_yolo2x.h"
#include "box.h"

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <math.h>
#include <float.h>
#include <string.h>

static int entry_index(struct region_layer *layer, int batch, int location, int entry)
{
    int n =   location / (layer->w*layer->h);
    int loc = location % (layer->w*layer->h);
    // leo:no batch
    return /*batch*l.outputs +*/ n*layer->w*layer->h*(layer->coords+layer->classes+1) + entry*layer->w*layer->h + loc;
}

//https://github.com/duangenquan/YoloV2NCS/blob/master/src/Region.cpp
typedef enum {
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
} ACTIVATION;
static inline float linear_activate(float x) { return x; }
static inline float logistic_activate(float x) { return 1. / (1. + exp(-x)); }
static inline float loggy_activate(float x) { return 2. / (1. + exp(-x)) - 1; }
static inline float relu_activate(float x) { return x*(x > 0); }
static inline float elu_activate(float x) { return (x >= 0)*x + (x < 0)*(exp(x) - 1); }
static inline float relie_activate(float x) { return (x > 0) ? x : .01*x; }
static inline float ramp_activate(float x) { return x*(x > 0) + .1*x; }
static inline float leaky_activate(float x) { return (x > 0) ? x : .1*x; }
static inline float tanh_activate(float x) { return (exp(2 * x) - 1) / (exp(2 * x) + 1); }

static inline float plse_activate(float x)
{
    if (x < -4) {
        return .01 * (x + 4);
    }
    if (x > 4) {
        return .01 * (x - 4) + 1;
    }
    return .125*x + .5;
}

#define ACTIVATE_FUNC(x)  logistic_activate(x)

#define DOABS 1

static box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
    box b;
    b.x = (i + ACTIVATE_FUNC(x[index + 0])) / w;
    b.y = (j + ACTIVATE_FUNC(x[index + 1])) / h;
    b.w = exp(x[index + 2]) * biases[2*n];
    b.h = exp(x[index + 3]) * biases[2*n+1];
    if(DOABS) {
        b.w = exp(x[index + 2]) * biases[2*n]   / w;
        b.h = exp(x[index + 3]) * biases[2*n+1] / h;
    }
    return b;
}

static void softmax_with_stride(float *input, int n, float temp, float *output, int stride)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for (i = 0; i < n; ++i) {
        if (input[i*stride] > largest) {
            largest = input[i*stride];
        }
    }
    for (i = 0; i < n; ++i) {
        float e = exp(input[i*stride] / temp - largest / temp);
        sum += e;
        output[i*stride] = e;
    }
    for (i = 0; i < n; ++i) {
        output[i*stride] /= sum;
    }
}

static void softmax(float *input, int n, float temp, float *output)
{
    int i;
    float sum = 0;
    float largest = input[0];
    for (i = 0; i < n; ++i) {
        if (input[i] > largest) {
            largest = input[i];
        }
    }
    for (i = 0; i < n; ++i) {
        float e = exp(input[i]/temp - largest/temp);
        sum += e;
        output[i] = e;
    }
    for (i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}

static float get_yolov2_get_max_val(float *val, int u32Num, int * max_value_idx)
{
    int i = 0;
    float max_value = 0;

    max_value = val[0];
    *max_value_idx = 0;
    for(i = 1; i < u32Num; i++) {
        if(val[i] > max_value) {
            max_value = val[i];
            *max_value_idx = i;
        }
    }

    return max_value;
}

static void get_region_boxes(struct region_layer *layer, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map)
{
    int i,j,n;
    float *predictions = layer->output;
    for (i = 0; i < layer->w*layer->h; ++i) {
        int row = i / layer->w;
        int col = i % layer->w;
        for(n = 0; n < layer->n; ++n) {
            int index = i*layer->n + n;
            int p_index = index * (layer->classes + 5) + 4;
            float scale = predictions[p_index];
            if(layer->classfix == -1 && scale < .5) {
                scale = 0;
            }
            int box_index = index * (layer->classes + 5);
            boxes[index] = get_region_box(predictions, layer->biases, n, box_index, col, row, layer->w, layer->h);
            boxes[index].x *= w;
            boxes[index].y *= h;
            boxes[index].w *= w;
            boxes[index].h *= h;
            //printf("get_region_boxes: n:%d box_idx:%d row:%d col:%d layer_w:%d, layer_h:%d\n", n, box_index, col, row, layer->w, layer->h);
            //printf("                : x:%f y:%f w:%f h:%f\n", boxes[index].x, boxes[index].y, boxes[index].w, boxes[index].h);


            int class_index = index * (layer->classes + 5) + 5;
            for(j = 0; j < layer->classes; ++j) {
                float prob = scale*predictions[class_index+j];
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
            if(only_objectness) {
                probs[index][0] = scale;
            }
        }
    }
}

static void free_ptrs(void **ptrs, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        free(ptrs[i]);
    }
    free(ptrs);
}

static void custom_get_region_detections(struct region_layer *layer, int w, int h, int net_w, int net_h, float thresh, int *map, float hier, int relative, detection *dets, int letter)
{
    //printf("custom_get_region_detections w:%d,h:%d,netw:%d,neth:%d thresh:%f,map:%p,hier:%f,relative:%d,letter:%d\n",
    //		w,h,net_w,net_h,thresh,map,hier,relative,letter);
    box *boxes = (box *)calloc(layer->w*layer->h*layer->n, sizeof(box));
    float **probs = (float **)calloc(layer->w*layer->h*layer->n, sizeof(float *));
    int i, j;
    for (j = 0; j < layer->w*layer->h*layer->n; ++j) {
        probs[j] = (float *)calloc(layer->classes, sizeof(float *));
    }
    get_region_boxes(layer, 1, 1, thresh, probs, boxes, 0, map);
    for (j = 0; j < layer->w*layer->h*layer->n; ++j) {
        dets[j].classes = layer->classes;
        dets[j].bbox = boxes[j];
        dets[j].objectness = 1;
        for (i = 0; i < layer->classes; ++i) {
            dets[j].prob[i] = probs[j][i];
        }
    }

    free(boxes);
    free_ptrs((void **)probs, layer->w*layer->h*layer->n);
}

static void fill_network_boxes(struct region_layer *layer, int w, int h, float thresh, float hier, int *map, int relative, detection *dets, int letter)
{
    int j;
#if 0
    for(j = 0; j < net->n; ++j) {
        layer l = net->layers[j];
        if(l.type == YOLO) {
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
            dets += count;
        }
        if(l.type == REGION) {
            get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
            dets += l.w*l.h*l.n;
        }
        if(l.type == DETECTION) {
            get_detection_detections(l, w, h, thresh, dets);
            dets += l.w*l.h*l.n;
        }
    }
#endif
    if(layer->type == LAYER_REGION) {
        custom_get_region_detections(layer, w, h, layer->net_w, layer->net_h, thresh, map, hier, relative, dets, letter);
        dets += layer->w*layer->h*layer->n;
    }
    // not ready
    // if(layer->type == DETECTION) {
    // 	get_detection_detections(l, w, h, thresh, dets);
    // 	dets += layer->w*layer->h*layer->n;
    // }
}

static int num_detections(struct region_layer *layer, float thresh)
{
    int i;
    int s = 0;
    s += layer->w*layer->h*layer->n;
#if 0
    for(i = 0; i < net->n; ++i) {
        layer l = net->layers[i];
        if(l.type == YOLO) {
            s += yolo_num_detections(l, thresh);
        }
        if(l.type == DETECTION || l.type == REGION) {
            s += l.w*l.h*l.n;
        }
    }
#endif
    return s;
}

static detection *make_network_boxes(struct region_layer *layer, float thresh, int *num)
{
    int i;
    int nboxes = num_detections(layer, thresh);
    if (num) {
        *num = nboxes;
    }
    printf("%s nboxes:%d\n", __FUNCTION__, nboxes);
    detection *dets = (detection *) calloc(nboxes, sizeof(detection));
    for(i = 0; i < nboxes; ++i) {
        dets[i].prob = (float*) calloc(layer->classes, sizeof(float));
        if(layer->coords > 4) {
            dets[i].mask = (float*)calloc(layer->coords-4, sizeof(float));
        }
    }
    return dets;
}

static void flatten(float *x, int size, int layers, int batch, int forward)
{
    float *swap = (float *)calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b) {
        for(c = 0; c < layers; ++c) {
            for(i = 0; i < size; ++i) {
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) {
                    swap[i2] = x[i1];
                } else {
                    swap[i1] = x[i2];
                }
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

static void forward_region_layer(struct region_layer *layer)
{
    int i,j,b,t,n;
    int size = layer->coords + layer->classes + 1;
    flatten(layer->output, layer->w*layer->h, size*layer->n, layer->batch, 1);
    for (b = 0; b < layer->batch; ++b) {
        for(i = 0; i < layer->h*layer->w*layer->n; ++i) {
            int index = size*i + b*layer->outputs;
            layer->output[index + 4] = logistic_activate(layer->output[index + 4]);
        }
    }
    for (b = 0; b < layer->batch; ++b) {
        for (i = 0; i < layer->h*layer->w*layer->n; ++i) {
            int index = size*i + b*layer->outputs;
            softmax_with_stride(layer->output + index + 5, layer->classes, 1, layer->output + index + 5, 1);
        }
    }
}

detection *imvt_yolo2x_get_detection(struct region_layer *layer, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter)
{
    forward_region_layer(layer);
    detection *dets = make_network_boxes(layer, thresh, num);
    fill_network_boxes(layer, w, h, thresh, hier, map, relative, dets, letter);
    return dets;
}

static int nms_comparator_v3(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if (b.sort_class >= 0) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0) {
        return 1;
    } else if (diff > 0) {
        return -1;
    }
    return 0;
}

void imvt_yolo2x_do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator_v3);
        for (i = 0; i < total; ++i) {
            //printf("  k = %d, \t i = %d \n", k, i);
            if (dets[i].prob[k] == 0) {
                continue;
            }
            box a = dets[i].bbox;
            for (j = i + 1; j < total; ++j) {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

void imvt_yolo2x_free_detection(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i) {
        free(dets[i].prob);
        if(dets[i].mask) {
            free(dets[i].mask);
        }
    }
    free(dets);
}

extern "C" {

    detection *imvt_yolo2x_get_detection_c(struct region_layer *layer, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter)
    {
        return imvt_yolo2x_get_detection(layer, w, h, thresh, hier, map, relative, num, letter);
    }

    void imvt_yolo2x_free_detection_c(detection *dets, int n)
    {
        imvt_yolo2x_free_detection(dets, n);
    }

    void imvt_yolo2x_do_nms_sort_c(detection *dets, int total, int classes, float thresh)
    {
        imvt_yolo2x_do_nms_sort(dets, total, classes, thresh);
    }

}