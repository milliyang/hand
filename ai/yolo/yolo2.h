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

class Yolo2
{
public:
    Yolo2(const string &model_file, const string &trained_file, const string &mean_file, const string &label_file);
    std::vector<YoloBox> Run(const cv::Mat &img, int N = 5);

    static std::string id_to_name(int id);

private:
    void SetMean(const string &mean_file);
    void SetMean(const string& mean_file, const string& mean_value);

    void WrapInputLayer(std::vector<cv::Mat> *input_channels);
    void Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels);
    void Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels, double normalize_value);
    void CaffeProcess(const cv::Mat &img);
    void PostProcess(const cv::Mat &img);
    void SelectYoloBox(detection *dets, int nbox);

private:
    shared_ptr<Net<float>> net_;
    cv::Size caffe_input_image_size_;
    int num_channels_;
    cv::Mat mean_;
    std::vector<YoloBox> yolo_boxs_;

    //
    int same_scale_factor;

    //layer output information
    float confidence_threshold_;
    float nms_threshold_;
    int grid_;
    int bbox_each_grid;
    int classes_;
    int total_bbox_;
    const char **class_names_;
    struct region_layer layer;

    //temp variable
    cv::Size origin_image_size_;
    //int origin_image_width_;
    //int origin_image_height_;
};
