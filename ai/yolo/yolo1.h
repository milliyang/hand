#pragma once

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "yolo.h"
#include "caffe_static_lib_common.h"

using namespace caffe;
using std::string;

class Yolo1
{
public:
    Yolo1(const string &model_file, const string &trained_file, const string &mean_file, const string &label_file);
    std::vector<YoloBox> Run(const cv::Mat &img, int N = 5);

private:
    void SetMean(const string &mean_file);

    void WrapInputLayer(std::vector<cv::Mat> *input_channels);
    void Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels);
    void CaffeProcess(const cv::Mat &img);
    void PostProcess(const cv::Mat &img);
    void SelectYoloBox(box *boxes, float **probs);

private:
    shared_ptr<Net<float>> net_;
    cv::Size caffe_input_image_size_;
    int num_channels_;
    cv::Mat mean_;
    std::vector<YoloBox> yolo_boxs_;

    //layer output information
    float confidence_threhold_;
    int grid_;
    int bbox_each_grid;
    int classes_;
    int total_bbox_;

    //temp variable
    cv::Size origin_image_size_;
    //int origin_image_width_;
    //int origin_image_height_;
};
