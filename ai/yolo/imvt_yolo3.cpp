//#include "stdafx.h"
#include "imvt_yolo3.h"
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

#define OLD_CODE_DYNAMIC_ALLOC_MEMPRY (1)

enum ACTIVATION {
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
};

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

static int entry_index2(struct region_layer *layer, int batch, int location, int entry)
{
    int n =   location / (layer->w*layer->h);
    int loc = location % (layer->w*layer->h);
    //printf("kao n:%d loc:%d w:%d h:%d coords:%d cls:%d entry:%d ret:%d\n", n, loc, layer->w, layer->h, layer->coords, layer->classes, entry, n*layer->w*layer->h*(layer->coords+layer->classes+1) + entry*layer->w*layer->h + loc);
    // leo:no batch
    return /*batch*l.outputs +*/ n*layer->w*layer->h*(layer->coords+layer->classes+1) + entry*layer->w*layer->h + loc;
}

static int entry_index(struct region_layer *layer, int batch, int location, int entry)
{
    int n =   location / (layer->w*layer->h);
    int loc = location % (layer->w*layer->h);

    //printf("kao n:%d loc:%d w:%d h:%d coords:%d cls:%d entry:%d ret:%d\n", n, loc, layer->w, layer->h, layer->coords, layer->classes, entry, n*layer->w*layer->h*(layer->coords+layer->classes+1) + entry*layer->w*layer->h + loc);
    // leo:no batch
    return /*batch*l.outputs +*/ n*layer->w*layer->h*(layer->coords+layer->classes+1) + entry*layer->w*layer->h + loc;
}

#if 0
box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0 * stride]) / w;
    b.y = (j + x[index + 1 * stride]) / h;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}
#else
//https://github.com/duangenquan/YoloV2NCS/blob/master/src/Region.cpp

static inline float linear_activate(float x)
{
    return x;
}
static inline float logistic_activate(float x)
{
    return 1. / (1. + exp(-x));
}
static inline float loggy_activate(float x)
{
    return 2. / (1. + exp(-x)) - 1;
}
static inline float relu_activate(float x)
{
    return x*(x > 0);
}
static inline float elu_activate(float x)
{
    return (x >= 0)*x + (x < 0)*(exp(x) - 1);
}
static inline float relie_activate(float x)
{
    return (x > 0) ? x : .01*x;
}
static inline float ramp_activate(float x)
{
    return x*(x > 0) + .1*x;
}
static inline float leaky_activate(float x)
{
    return (x > 0) ? x : .1*x;
}
static inline float tanh_activate(float x)
{
    return (exp(2 * x) - 1) / (exp(2 * x) + 1);
}
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
static inline float lhtan_activate(float x)
{
    if (x < 0) {
        return .001*x;
    }
    if (x > 1) {
        return .001*(x - 1) + 1;
    }
    return x;
}
static inline float stair_activate(float x)
{
    int n = floor(x);
    if (n % 2 == 0) {
        return floor(x / 2.);
    } else {
        return (x - n) + floor(x / 2.);
    }
}
static inline float hardtan_activate(float x)
{
    if (x < -1) {
        return -1;
    }
    if (x > 1) {
        return 1;
    }
    return x;
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

#if 1
static float activate(float x, ACTIVATION a)
{
    switch (a) {
    case LINEAR:
        return linear_activate(x);
    case LOGISTIC:
        return logistic_activate(x);
    case LOGGY:
        return loggy_activate(x);
    case RELU:
        return relu_activate(x);
    case ELU:
        return elu_activate(x);
    case RELIE:
        return relie_activate(x);
    case RAMP:
        return ramp_activate(x);
    case LEAKY:
        return leaky_activate(x);
    case TANH:
        return tanh_activate(x);
    case PLSE:
        return plse_activate(x);
    case STAIR:
        return stair_activate(x);
    case HARDTAN:
        return hardtan_activate(x);
    case LHTAN:
        return lhtan_activate(x);
    }
    return 0;
}
#endif

static void activate_array(float *x, const int n, ACTIVATION a)
{
    int i;
    for (i = 0; i < n; ++i) {
        x[i] = activate(x[i], a);
    }
}

static void forward_yolo_layer_for_activation(struct region_layer *layer)
{
    int i, j, b, t, n;
    b = 0;
    //for (b = 0; b < layer->batch; ++b) {
    for (n = 0; n < layer->n; ++n) {
        int index = entry_index(layer, b, n*layer->w*layer->h, 0);
        activate_array(layer->output + index, 2 * layer->w*layer->h, LOGISTIC);
        //printf("activate_array index:%d num:%d activation:%d\n", index, 2 * layer->w*layer->h, LOGISTIC);

        index = entry_index(layer, b, n*layer->w*layer->h, 4);
        activate_array(layer->output + index, (1 + layer->classes)*layer->w*layer->h, LOGISTIC);
        //printf("activate_array index:%d num:%d activation:%d\n", index, (1 + layer->classes)*layer->w*layer->h, LOGISTIC);
    }
    //}

    // printf("layer>>\n");
    // printf(" ->type:%d\n", layer->type);
    // printf(" ->n:%d\n", layer->n);
    // printf(" ->w:%d\n", layer->w);
    // printf(" ->h:%d\n", layer->h);
    // printf(" ->batch:%d\n", layer->batch);
    // //printf(" ->biases[20]:%d\n", layer->biases[20]);
    // printf(" ->classes:%d\n", layer->classes);
    // printf(" ->classfix:%d\n", layer->classfix);
    // printf(" ->coords:%d\n", layer->coords);
    // printf(" ->background:%d\n", layer->background);
    // printf(" ->net_w:%d\n", layer->net_w);
    // printf(" ->net_h:%d\n", layer->net_h);

    // for (int i = 0; i < 100; i++) {
    //     printf("forward_yolo_layer output[%d]:%f\n", i, layer->output[i]);
    // }
}

static int yolo_num_detections(struct region_layer *layer, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < layer->w*layer->h; ++i) {
        for(n = 0; n < layer->n; ++n) {
            int obj_index  = entry_index2(layer, 0, n*layer->w*layer->h + i, 4);
            if(layer->output[obj_index] > thresh) {
                ++count;
            }
        }
    }
    return count;
}

static int num_detections(struct region_layer *layer, float thresh)
{
    int i;
    int s = 0;
#if 0
    s += yolo_num_detections(layer, thresh);
    printf("yolo_num_detections num:%d thresh:%f\n", s, thresh);
#else
    if (layer->type == LAYER_YOLO) {
        for (i = 0; i < layer->sub_layer_num; i++) {
            layer->w = layer->sub_layer[i].w;
            layer->h = layer->sub_layer[i].h;
            layer->mask[0] = layer->sub_layer[i].mask[0];
            layer->mask[1] = layer->sub_layer[i].mask[1];
            layer->mask[2] = layer->sub_layer[i].mask[2];
            layer->output = layer->sub_layer[i].output;
            s += yolo_num_detections(layer, thresh);
            //printf("yolo_num_detections num:%d thresh:%f\n", s, thresh);
        }
    } else {
        s += layer->w*layer->h*layer->n;
        //printf("yolo_num_detections num:%d thresh:%f\n", s, thresh);
    }
#endif

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
#if OLD_CODE_DYNAMIC_ALLOC_MEMPRY
    if (num) {
        *num = nboxes;
    }
    dets = (detection *) calloc(nboxes, sizeof(detection));
    for(int i = 0; i < nboxes; ++i) {
        dets[i].prob = (float*) calloc(layer->classes, sizeof(float));
        if(layer->coords > 4) {
            dets[i].mask = (float*)calloc(layer->coords-4, sizeof(float));
        }
    }
    return dets;
#else
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
#endif
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

static box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

static void correct_yolo_boxes_v3(detection *dets, int n, int w, int h, int netw, int neth, int relative)
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

static void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    new_w = netw;
    new_h = neth;
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

static int get_yolo_detections(struct region_layer *layer, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = layer->output;
    int *input;
    //if (layer->batch == 2) avg_flipped_yolo(layer);
#if 0
    if (s_yolo_ctx.quant_base == IMVT_YOLO_QUANT_BASE_HISI) {
        input = (int *)predictions;
        for (int i = 0; i < layer->outputs; i++) {
            predictions[i] = (float)(input[i]) / IMVT_YOLO_QUANT_BASE_HISI;
        }
    }
#endif
    int count = 0;
    for (i = 0; i < layer->w*layer->h; ++i) {
        int row = i / layer->w;
        int col = i % layer->w;
        for(n = 0; n < layer->n; ++n) {
            int obj_index  = entry_index(layer, 0, n*layer->w*layer->h + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) {
                continue;
            }
            int box_index  = entry_index(layer, 0, n*layer->w*layer->h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, layer->biases, layer->mask[n], box_index, col, row, layer->w, layer->h, netw, neth, layer->w*layer->h);

#if 0
            printf("get_yolo obj_index:%d count:%d box_index:%d box:%f %f %f %f n:%d mask:%d caffe\n",
                   obj_index, count, box_index,
                   dets[count].bbox.x, dets[count].bbox.y, dets[count].bbox.w, dets[count].bbox.h,
                   n,
                   layer->mask[n]);
#endif
            dets[count].objectness = objectness;
            dets[count].classes = layer->classes;
            for(j = 0; j < layer->classes; ++j) {
                int class_index = entry_index(layer, 0, n*layer->w*layer->h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    //printf("w:%d h:%d netw:%d neth:%d relative:%d\n", w, h, netw, neth, relative);
    correct_yolo_boxes_v3(dets, count, w, h, netw, neth, relative);
    return count;
}

static void fill_network_boxes(struct region_layer *layer, int w, int h, float thresh, float hier, int *map, int relative, detection *dets)
{
    int j, count;
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
    } else if(layer->type == LAYER_YOLO) {
        //TODO:
        // layer->net_w, layer->net_h maybe should change to 3 layer
        for (int i = 0; i < layer->sub_layer_num; i++) {
            layer->w = layer->sub_layer[i].w;
            layer->h = layer->sub_layer[i].h;
            layer->mask[0] = layer->sub_layer[i].mask[0];
            layer->mask[1] = layer->sub_layer[i].mask[1];
            layer->mask[2] = layer->sub_layer[i].mask[2];
            layer->output = layer->sub_layer[i].output;
            count = get_yolo_detections(layer, w, h, layer->net_w, layer->net_h, thresh, map, relative, dets);
            dets += count;
        }
    }
    // not ready
    // if(layer->type == DETECTION) {
    // 	get_detection_detections(l, w, h, thresh, dets);
    // 	dets += layer->w*layer->h*layer->n;
    // }
}

detection *imvt_yolo3_get_detection(struct region_layer *layer, int w, int h, float thresh, float hier, int *map, int relative, int *num)
{
    int *input;
    if (layer->type == LAYER_YOLO) {
        if (layer->sub_layer_continue_memory) {
            layer->sub_layer[0].output = layer->output;
            layer->sub_layer[1].output = layer->output + layer->sub_layer[0].outputs;
            layer->sub_layer[2].output = layer->output + layer->sub_layer[0].outputs + layer->sub_layer[1].outputs;
        }
        for (int i = 0; i < layer->sub_layer_num; i++) {
            layer->w = layer->sub_layer[i].w;
            layer->h = layer->sub_layer[i].h;
            layer->mask[0] = layer->sub_layer[i].mask[0];
            layer->mask[1] = layer->sub_layer[i].mask[1];
            layer->mask[2] = layer->sub_layer[i].mask[2];
            layer->output = layer->sub_layer[i].output;
            layer->outputs = layer->sub_layer[i].outputs;

            if (s_yolo_ctx.quant_base == IMVT_YOLO_QUANT_BASE_HISI) {
                input = (int *)layer->output;
                for (int i = 0; i < layer->outputs; i++) {
                    layer->output[i] = (float)(input[i]) / IMVT_YOLO_QUANT_BASE_HISI;
                }
            }
            forward_yolo_layer_for_activation(layer);
        }
    }
    detection *dets = make_network_boxes(layer, thresh, num);

    fill_network_boxes(layer, w, h, thresh, hier, map, relative, dets);
    // int j;
    // for (j = 0; j < *num; j++) {
    // box b = dets[j].bbox;
    // printf("all box[%d]:%f %f %f %f caffe\n", j, b.x, b.y, b.w, b.h);
    // }
    return dets;
}

void imvt_yolo3_do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    do_nms_sort(dets, total, classes, thresh);
}

void imvt_yolo3_free_detection(detection *dets, int n)
{
#if OLD_CODE_DYNAMIC_ALLOC_MEMPRY
    int i;
    for (i = 0; i < n; ++i) {
        free(dets[i].prob);
        if (dets[i].mask) {
            free(dets[i].mask);
        }
    }
    free(dets);
#else
    return;
#endif
}

int imvt_yolo3_deinit(void)
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

int imvt_yolo3_init(struct region_layer *layer, void* mem_buffer, int size, int quant_base)
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

#if OLD_CODE_DYNAMIC_ALLOC_MEMPRY
    s_yolo_ctx.copy_net_output = 0;
    s_yolo_ctx.p_buffer = NULL;
    s_yolo_ctx.buffer_size = 0;

    s_yolo_ctx.p_output = NULL;
    s_yolo_ctx.p_detection = 0;
#else
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
#endif
    return OK;
}

extern "C" {

    detection *imvt_yolo3_get_detection_c(struct region_layer *layer, int w, int h, float thresh, float hier, int *map, int relative, int *num)
    {
        return imvt_yolo3_get_detection(layer, w, h, thresh, hier, map, relative, num);
    }

    void imvt_yolo3_free_detection_c(detection *dets, int n)
    {
        imvt_yolo3_free_detection(dets, n);
    }

    void imvt_yolo3_do_nms_sort_c(detection *dets, int total, int classes, float thresh)
    {
        do_nms_sort(dets, total, classes, thresh);
    }

    int imvt_yolo3_get_mmz_memory_size_c(struct region_layer *layer)
    {
        return _yolo2_get_memory_needed(layer, NULL, NULL);
    }
    int imvt_yolo3_init_c(struct region_layer *layer, void* mem_buffer, int size, int quant_base)
    {
        return imvt_yolo3_init(layer, mem_buffer, size, quant_base);
    }
    int imvt_yolo3_deinit_c(void)
    {
        return imvt_yolo3_deinit();
    }
}