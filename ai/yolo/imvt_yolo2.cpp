//#include "stdafx.h"
#include "imvt_yolo2.h"
#include "box.h"

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <math.h>
#include <assert.h>
#include <string.h>

#ifndef OK
#define OK (0)
#define NG (-1)
#endif

#define IMVT_YOLO_MEM_TYPE_SYS   (0)
#define IMVT_YOLO_MEM_TYPE_MMZ   (1)

struct bbox_t {
    unsigned int x, y, w, h;	// (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;					// confidence - probability that the object was found correctly
    unsigned int obj_id;		// class of object - from range [0, classes-1]
    unsigned int track_id;		// tracking id for video (0 - untracked, 1 - inf - tracked object)
};

struct imvt_yolo_ctx {
    int mem_type;
    int copy_net_output;
    int quant_base;
    unsigned char *p_buffer;
    int buffer_size;

    float *p_output;
    unsigned char *p_detection;

    int net_output_len;
    int detection_len;
};
static struct imvt_yolo_ctx s_yolo_ctx;


static int max_index(float *a, int n)
{
    if(n <= 0) {
        return -1;
    }
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i) {
        if(a[i] > max) {
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
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
    detection *dets = NULL;
    int detection_len;
    unsigned char *last;
    int nboxes = num_detections(layer, thresh);
    if (num) {
        *num = nboxes;
    }
#if 0
    // old code
    if (num) {
        *num = nboxes;
    }
    detection *dets = (detection *) calloc(nboxes, sizeof(detection));
    for(int i = 0; i < nboxes; ++i) {
        dets[i].prob = (float*) calloc(layer->classes, sizeof(float));
        if(layer->coords > 4) {
            dets[i].mask = (float*)calloc(layer->coords-4, sizeof(float));
        }
    }
#endif

    assert(s_yolo_ctx.p_buffer != NULL);

    //pre-alloc memory
    dets = (detection *) s_yolo_ctx.p_detection;
    last = s_yolo_ctx.p_detection + (nboxes * sizeof(detection));

    for(int i = 0; i < nboxes; ++i) {
        dets[i].prob = (float*)last;
        last += layer->classes * sizeof(float);
        if(layer->coords > 4) {
            dets[i].mask = (float*) last;
            last += (layer->coords - 4) * sizeof(float);
        }
    }
    return dets;
}

static void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i) {
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

static int entry_index(struct region_layer *layer, int batch, int location, int entry)
{
    int n =   location / (layer->w*layer->h);
    int loc = location % (layer->w*layer->h);
    // leo:no batch
    return /*batch*l.outputs +*/ n*layer->w*layer->h*(layer->coords+layer->classes+1) + entry*layer->w*layer->h + loc;
}

#if 0
box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}
#else
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

static box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = ((float)i + ACTIVATE_FUNC(x[index + 0 * stride])) / (float)w;
    b.y = ((float)j + ACTIVATE_FUNC(x[index + 1 * stride])) / (float)h;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / (float)w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / (float)h;
    return b;
}
#endif


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

static void get_region_detections(struct region_layer *layer, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets)
{
    int i,j,n,z;
    float *predictions;
    int *input;

    if (s_yolo_ctx.copy_net_output) {
        memcpy(s_yolo_ctx.p_output, layer->output, layer->outputs * sizeof(float));
        predictions = s_yolo_ctx.p_output;
    } else {
        predictions = layer->output;
    }
    if (s_yolo_ctx.quant_base == IMVT_YOLO_QUANT_BASE_HISI) {
        input = (int *)predictions;
        for(int i = 0; i < layer->outputs; i++) {
            predictions[i] = (float)(input[i]) / IMVT_YOLO_QUANT_BASE_HISI;
        }
    }

    float softmax_prob[80];
    float prob;

    int max_prob;
    int max_prob_idx;

    for (i = 0; i < layer->w*layer->h; ++i) {
        int row = i / layer->w;
        int col = i % layer->w;
        for(n = 0; n < layer->n; ++n) {
            int index = n*layer->w*layer->h + i;
            for(j = 0; j < layer->classes; ++j) {
                dets[index].prob[j] = 0;
            }
            int obj_index  = entry_index(layer, 0, n*layer->w*layer->h + i, layer->coords);
            int box_index  = entry_index(layer, 0, n*layer->w*layer->h + i, 0);
            int mask_index = entry_index(layer, 0, n*layer->w*layer->h + i, 4);
            float scale = layer->background ? 1 : predictions[obj_index];

            //printf("row:%d col:%d obj_index:%d box_index:%d mask_index:%d\n", row, col, obj_index, box_index, mask_index);

            scale = ACTIVATE_FUNC(scale);

            //printf("i:%d n:%d obj_index:%d predictions:%f scale:%f sigmiod:%f\n", i, n, obj_index, predictions[obj_index], scale, SAMPLE_SVP_NNIE_SIGMOID(scale));

            dets[index].bbox = get_region_box(predictions, layer->biases, n, box_index, col, row, layer->w, layer->h, layer->w*layer->h);
            dets[index].objectness = scale > thresh ? scale : 0;
            if(dets[index].mask) {
                for(j = 0; j < layer->coords - 4; ++j) {
                    dets[index].mask[j] = layer->output[mask_index + j*layer->w*layer->h];
                }
            }
            if (dets[index].objectness) {
#if 0
                for (j = 0; j < layer->classes; ++j) {
                    int class_index = entry_index(layer, 0, n*layer->w*layer->h + i, layer->coords + 1 + j);

                    //float prob = scale*predictions[class_index];
                    float prob = scale* predictions[class_index];
                    dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    if (prob > thresh) {
                        printf("i:%d n:%d j:%d scale:%f prob:%f class_index:%d\n", i, n, j, scale, prob, class_index);
                    }
                }
#else
                for (j = 0; j < layer->classes; ++j) {
                    int class_index = entry_index(layer, 0, n*layer->w*layer->h + i, layer->coords + 1 + j);
                    softmax_prob[j] = predictions[class_index];
                }
                softmax(softmax_prob, layer->classes, 1, softmax_prob);

                max_prob = get_yolov2_get_max_val(softmax_prob, layer->classes, &max_prob_idx);

                // for(j = 0; j < layer->classes; ++j) {
                // 	prob = softmax_prob[j] * scale;
                // 	printf("i:%d n:%d j:%d scale:%f prob:%f class_index:%d\n", i, n, j, scale, prob, j);
                // 	if (j == max_prob_idx) {
                // 		printf("MAX!!!\n");
                // 		dets[index].prob[j] = (prob > thresh) ? prob : 0;
                // 	} else {
                // 		dets[index].prob[j] = 0;
                // 	}
                // }

                for(j = 0; j < layer->classes; ++j) {
                    prob = softmax_prob[j] * scale;
                    dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    // if (prob > thresh) {
                    //     printf("i:%d n:%d j:%d scale:%f prob:%f class_index:%d\n", i, n, j, scale, prob, j);
                    // }
                }
#endif

            }
        }
    }
    correct_region_boxes(dets, layer->w*layer->h*layer->n, w, h, netw, neth, relative);
}

static void fill_network_boxes(struct region_layer *layer, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
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
        get_region_detections(layer, w, h, layer->net_w, layer->net_h, thresh, map, hier, relative, dets);
        dets += layer->w*layer->h*layer->n;
    }
    // not ready
    // if(layer->type == DETECTION) {
    // 	get_detection_detections(l, w, h, thresh, dets);
    // 	dets += layer->w*layer->h*layer->n;
    // }
}

detection *imvt_yolo2_get_detection(struct region_layer *layer, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    detection *dets = make_network_boxes(layer, thresh, num);
    fill_network_boxes(layer, w, h, thresh, hier, map, relative, dets);
    return dets;
}

void imvt_yolo2_do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    do_nms_sort(dets, total, classes, thresh);
}

void imvt_yolo2_free_detection(detection *dets, int n)
{
    return;
#if 0
    int i;
    for (i = 0; i < n; ++i) {
        free(dets[i].prob);
        if (dets[i].mask) {
            free(dets[i].mask);
        }
    }

    //free(dets);
#endif
}

int imvt_yolo2_deinit(void)
{
    if (s_yolo_ctx.mem_type == IMVT_YOLO_MEM_TYPE_SYS && s_yolo_ctx.p_buffer != NULL) {
        free(s_yolo_ctx.p_buffer);
        s_yolo_ctx.p_buffer = NULL;
        s_yolo_ctx.p_output = NULL;
        s_yolo_ctx.p_detection = NULL;
    }
    return OK;
}

static int _yolo2_get_memory_needed(struct region_layer *layer, int *len_net_output, int *len_dets)
{
    int detection_len = 0;
    int net_output_size = 0;
    //return how many memory needed will be alloc

    //1. detection
    int nboxes = num_detections(layer, 0.0f);
    detection_len += nboxes * sizeof(detection);
    for(int i = 0; i < nboxes; ++i) {
        detection_len += layer->classes * sizeof(float);
        if(layer->coords > 4) {
            detection_len += (layer->coords - 4) * sizeof(float);
        }
    }

    //2. network outputs
    //layer.outputs = layer.w * layer.h * layer.n * (layer.coords + 1 + layer.classes);
    net_output_size = layer->outputs * sizeof(float);

    //printf("%s detection_len:%d\n", __FUNCTION__, detection_len);
    //printf("%s net_output_size:%d\n", __FUNCTION__, net_output_size);
    if (len_net_output != NULL) {
        *len_net_output = net_output_size;
    }
    if (len_dets != NULL) {
        *len_dets = detection_len;
    }
    return net_output_size + detection_len;
}

int imvt_yolo2_init(struct region_layer *layer, void* mem_buffer, int size, int quant_base)
{
    int needed;
    memset(&s_yolo_ctx, 0, sizeof(s_yolo_ctx));
    // s_yolo_ctx.mem_type = IMVT_YOLO_MEM_TYPE_SYS;
    // s_yolo_ctx.p_buffer = NULL;
    s_yolo_ctx.quant_base = quant_base;
    if (quant_base == IMVT_YOLO_QUANT_BASE_HISI) {
        s_yolo_ctx.copy_net_output = 1;
    } else {
        s_yolo_ctx.copy_net_output = 0;
    }

    needed = _yolo2_get_memory_needed(layer, &s_yolo_ctx.net_output_len, &s_yolo_ctx.detection_len);

    //allocte the memory or use the pre-allocated MMZ memory
    if (mem_buffer != NULL) {
        if (size < needed) {
            return NG;
        } else {
            s_yolo_ctx.p_buffer = (unsigned char*)mem_buffer;
            s_yolo_ctx.buffer_size = size;

            s_yolo_ctx.p_output = (float*) s_yolo_ctx.p_buffer;
            s_yolo_ctx.p_detection = s_yolo_ctx.p_buffer + s_yolo_ctx.net_output_len;
        }
        s_yolo_ctx.mem_type = IMVT_YOLO_MEM_TYPE_MMZ;
    } else {
        size = needed;
        mem_buffer = calloc(1, needed);
        assert(mem_buffer != NULL);
        if (mem_buffer == NULL) {
            return NG;
        }
    }

    s_yolo_ctx.p_buffer = (unsigned char*)mem_buffer;
    s_yolo_ctx.buffer_size = size;

    s_yolo_ctx.p_output = (float*) s_yolo_ctx.p_buffer;
    s_yolo_ctx.p_detection = s_yolo_ctx.p_buffer + s_yolo_ctx.net_output_len;
    return OK;
}

extern "C" {

    detection *imvt_yolo2_get_detection_c(struct region_layer *layer, int w, int h, float thresh, float hier, int *map, int relative, int *num)
    {
        return imvt_yolo2_get_detection(layer, w, h, thresh, hier, map, relative, num);
    }

    void imvt_yolo2_free_detection_c(detection *dets, int n)
    {
        imvt_yolo2_free_detection(dets, n);
    }

    void imvt_yolo2_do_nms_sort_c(detection *dets, int total, int classes, float thresh)
    {
        do_nms_sort(dets, total, classes, thresh);
    }

    int imvt_yolo2_get_mmz_memory_size_c(struct region_layer *layer)
    {
        return _yolo2_get_memory_needed(layer, NULL, NULL);
    }
    int imvt_yolo2_init_c(struct region_layer *layer, void* mem_buffer, int size, int quant_base)
    {
        return imvt_yolo2_init(layer, mem_buffer, size, quant_base);
    }
    int imvt_yolo2_deinit_c(void)
    {
        return imvt_yolo2_deinit();
    }
}