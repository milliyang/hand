#pragma once

//Caffe
#include <caffe/caffe.hpp>
//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/common.hpp"
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/relu_layer.hpp"

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

//tiny yolo
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/bias_layer.hpp"

//For Yolov2
#include "caffe/layers/pass_through_layer.hpp"
#include "caffe/layers/reorg_layer.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/upsample_layer.hpp"
#include "caffe/layers/flatten_layer.hpp"

#ifdef CPU_ONLY
#define YOLO_CAFFE_MODE Caffe::CPU
#else
#define YOLO_CAFFE_MODE Caffe::GPU
#endif

namespace caffe
{
/*
	FixCompileError:
	F0326 17:28 : 37.700534 12680 layer_factory.cpp : 62] Check failed : registry.count(type) == 1 (0 vs. 1)
	Unknown layer type : BatchNorm(known types : Convolution, Dropout, Eltwise, InnerProduct, Input, LRN, PassThrough, Pooling, Power, Python, ReLU, Reorg, Sigmoid, Softmax, Split, TanH)
	*** Check failure stack trace : ***
*/
extern INSTANTIATE_CLASS(InputLayer);
extern INSTANTIATE_CLASS(InnerProductLayer);
extern INSTANTIATE_CLASS(DropoutLayer);
extern INSTANTIATE_CLASS(ConvolutionLayer);
extern INSTANTIATE_CLASS(ReLULayer);
extern INSTANTIATE_CLASS(PoolingLayer);
extern INSTANTIATE_CLASS(LRNLayer);
extern INSTANTIATE_CLASS(SoftmaxLayer);

extern INSTANTIATE_CLASS(BatchNormLayer);
extern INSTANTIATE_CLASS(ConcatLayer);

//tiny yolo
extern INSTANTIATE_CLASS(ScaleLayer);
extern INSTANTIATE_CLASS(BiasLayer);

//Yolov2
extern INSTANTIATE_CLASS(PassThroughLayer);
extern INSTANTIATE_CLASS(ReorgLayer);
extern INSTANTIATE_CLASS(ReshapeLayer);

//Yolov3
extern INSTANTIATE_CLASS(UpsampleLayer);
extern INSTANTIATE_CLASS(FlattenLayer);

#ifndef REGISTER_LAYER_CLASS_Convolution
#define REGISTER_LAYER_CLASS_Convolution
//REGISTER_LAYER_CLASS(Convolution);
#endif
#ifndef REGISTER_LAYER_CLASS_ReLU
#define REGISTER_LAYER_CLASS_ReLU
//REGISTER_LAYER_CLASS(ReLU);
#endif
#ifndef REGISTER_LAYER_CLASS_Pooling
#define REGISTER_LAYER_CLASS_Pooling
//REGISTER_LAYER_CLASS(Pooling);
#endif
#ifndef REGISTER_LAYER_CLASS_LRN
#define REGISTER_LAYER_CLASS_LRN
//REGISTER_LAYER_CLASS(LRN);
#endif
#ifndef REGISTER_LAYER_CLASS_Softmax
#define REGISTER_LAYER_CLASS_Softmax
//REGISTER_LAYER_CLASS(Softmax);
#endif
}
