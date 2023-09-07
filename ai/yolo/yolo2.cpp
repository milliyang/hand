//#include "stdafx.h"
#include "imvt_yolo2.h"
#include "imvt_yolo3.h"
#include "yolo2.h"
#include "box.h"

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

const static char *yolo2_class_name80[] = {
    //"background  ",
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
};

const static char *yolo2_class_name20[] = {
    "person",
    "face",
    "hand",
    "tvmonitor",
    "aeroplane",
    "chair",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "cow",
    "dog",
    "horse",
    "motorbike",
    "sheep",
    "sofa",
    "train",
};

int Yolo2::s_yolo2_classes;

std::string Yolo2::id_to_name(int id)
{
    if (s_yolo2_classes == 80) {
        if (id < s_yolo2_classes) {
            return std::string(yolo2_class_name80[id]);
        }
    } else {
        if (id < s_yolo2_classes) {
            return std::string(yolo2_class_name20[id]);
        }
    }
    return std::to_string(id);
}

Yolo2::Yolo2(const string &model_file, const string &trained_caffemodel, const string &mean_file, const string &label_file)
{
    Caffe::set_mode(YOLO_CAFFE_MODE);
    std::cerr << "mode is gpu:" << YOLO_CAFFE_MODE << std::endl;
    if (YOLO_CAFFE_MODE) {
        Caffe::DeviceQuery();
    }

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_caffemodel);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";


    Blob<float> *input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
    caffe_input_image_size_ = cv::Size(input_layer->width(), input_layer->height());

    std::cerr << "net name:" << net_->name() << std::endl;

    /* Load the binaryproto mean file. */
    //SetMean(mean_file);
    SetMean("", "0,0,0");

    Blob<float> *output_layer = net_->output_blobs()[0];
    std::cout << "output_layer->channels:" << output_layer->channels() << std::endl;
    // Blob<float> *output_layer1 = net_->output_blobs()[1];
    // std::cout << "output_layer1->channels:" << output_layer1->channels() << std::endl;
    // Blob<float> *output_layer2 = net_->output_blobs()[2];
    // std::cout << "output_layer2->channels:" << output_layer2->channels() << std::endl;

    // Blob<float> *output_layer = net_->output_blobs()[0];
    // CHECK_EQ(labels_.size(), output_layer->channels())
    // 	<< "Number of labels is different from the output layer dimension.";

    if (net_->name() == "yolov3" || net_->name() == "yolov3_mini") {
        layer.type = LAYER_YOLO;
        //CHECK_GE(net_->num_outputs(), 2) << "Network should have >= 2 output.";
    } else {
        layer.type = LAYER_REGION;
        CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";
    }
    std::cerr << "net name: " << net_->name() << " type: " << layer.type << std::endl;

    layer.batch = 1;

    if (net_->name() == "yolov2") {
        layer.classes = 80;
        s_yolo2_classes = 80;
        class_names_ = yolo2_class_name80;
    } else {
        layer.classes = 20;
        s_yolo2_classes = 20;
        class_names_ = yolo2_class_name20;
    }
    std::cerr << "yolo output class num:" << layer.classes << std::endl;

    layer.classfix = 1;
    layer.coords = 4;
    layer.background = 0;
    layer.net_w = 416;
    layer.net_h = 416;
    if (layer.type == LAYER_YOLO) {
        //mask = 0, 1, 2
        //anchors = 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326
        //classes = 80
        //num = 9
        //jitter = .3
        //ignore_thresh = .5
        //truth_thresh = 1
        //random = 1
        layer.biases[0] = 10;  //region_param.anchors:
        layer.biases[1] = 13;
        layer.biases[2] = 16;
        layer.biases[3] = 30;
        layer.biases[4] = 33;
        layer.biases[5] = 23;
        layer.biases[6] = 30;
        layer.biases[7] = 61;
        layer.biases[8] = 62;
        layer.biases[9] = 45;
        layer.biases[10] = 59;
        layer.biases[11] = 119;
        layer.biases[12] = 116;
        layer.biases[13] = 90;
        layer.biases[14] = 156;
        layer.biases[15] = 198;
        layer.biases[16] = 373;
        layer.biases[17] = 326;
        //layer     filters    size              input                output
        // 81 conv    255  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 255 0.088 BF   //forward_yolo_layer l.outputs:43095 n:3 w:13 h:13
        // 93 conv    255  1 x 1 / 1    26 x  26 x 512   ->    26 x  26 x 255 0.177 BF   //forward_yolo_layer l.outputs:172380 n:3 w:26 h:26
        //105 conv    255  1 x 1 / 1    52 x  52 x 256   ->    52 x  52 x 255 0.353 BF   //forward_yolo_layer l.outputs:689520 n:3 w:52 h:52
#define L1_WIDTH  (13)
#define L2_WIDTH  (26)
#define L3_WIDTH  (52)
        layer.n = 3;
        // layer.w = layer.h = L3_WIDTH;
        // layer.outputs = layer.w * layer.h * layer.n * (layer.coords + 1 + layer.classes);
        // layer.mask[0] = 0;
        // layer.mask[1] = 1;
        // layer.mask[2] = 2;
        layer.sub_layer_num = 3;
        layer.sub_layer_continue_memory = 1;
        layer.sub_layer[0].w = L1_WIDTH;
        layer.sub_layer[0].h = L1_WIDTH;
        layer.sub_layer[0].output = NULL;
        layer.sub_layer[0].outputs = L1_WIDTH * L1_WIDTH * layer.n * (layer.coords + 1 + layer.classes);
        layer.sub_layer[0].mask[0] = 6;
        layer.sub_layer[0].mask[1] = 7;
        layer.sub_layer[0].mask[2] = 8;

        layer.sub_layer[1].w = L2_WIDTH;
        layer.sub_layer[1].h = L2_WIDTH;
        layer.sub_layer[1].output = NULL;
        layer.sub_layer[1].outputs = L2_WIDTH * L2_WIDTH * layer.n * (layer.coords + 1 + layer.classes);
        layer.sub_layer[1].mask[0] = 3;
        layer.sub_layer[1].mask[1] = 4;
        layer.sub_layer[1].mask[2] = 5;

        layer.sub_layer[2].w = L3_WIDTH;
        layer.sub_layer[2].h = L3_WIDTH;
        layer.sub_layer[2].output = NULL;
        layer.sub_layer[2].outputs = L3_WIDTH * L3_WIDTH * layer.n * (layer.coords + 1 + layer.classes);
        layer.sub_layer[2].mask[0] = 0;
        layer.sub_layer[2].mask[1] = 1;
        layer.sub_layer[2].mask[2] = 2;
        confidence_threshold_ = 0.5;
        layer.outputs = layer.sub_layer[0].outputs;
    } else {
        confidence_threshold_ = 0.15;

        layer.sub_layer_num = 0;
        layer.sub_layer_continue_memory = 0;
        layer.w = layer.h = 13;
        layer.n = 5;
        layer.outputs = layer.w * layer.h * layer.n * (layer.coords + 1 + layer.classes);
        layer.sub_layer[0].outputs = layer.outputs;
        if (net_->name() == "yolov2") {
            layer.biases[0] = 0.57273; //region_param.anchors:
            layer.biases[1] = 0.677385;
            layer.biases[2] = 1.87446;
            layer.biases[3] = 2.06253;
            layer.biases[4] = 3.33843;
            layer.biases[5] = 5.47434;
            layer.biases[6] = 7.88282;
            layer.biases[7] = 3.52778;
            layer.biases[8] = 9.77052;
            layer.biases[9] = 9.16828;
            //layer     filters    size              input                output
            //30 conv    425  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 425 0.147 BF
            //layer.outputs=71994
        } else {
            layer.biases[0] = 1.3221;   //region_param.anchors:
            layer.biases[1] = 1.73145;
            layer.biases[2] = 3.19275;
            layer.biases[3] = 4.00944;
            layer.biases[4] = 5.05587;
            layer.biases[5] = 8.09892;
            layer.biases[6] = 9.47112;
            layer.biases[7] = 4.84053;
            layer.biases[8] = 11.2364;
            layer.biases[9] = 10.0071;
            //layer     filters    size              input                output
            //30 conv    125  1 x 1 / 1    13 x  13 x1024   ->    13 x  13 x 125 0.147 BF
            //layer.outputs=21125
        }
    }
    printf("layer.outputs:%d\n", layer.outputs);

    //layer output information
    same_scale_factor = 1;
    nms_threshold_ = 0.3;
    grid_ = layer.w;
    bbox_each_grid = layer.n;
    classes_ = layer.classes;
    total_bbox_ = grid_ * grid_ * bbox_each_grid; //13 * 13 * 5; anchor box
    //channel = ( 5 * (5+20) = 125)
    //eg.
    //每个box包含5个坐标值和20个类别，所以总共是5 * （5+20）= 125个输出维度
    //hisi:
    //每个box包含5个坐标值和5个类别，所以总共是5 * （5+5）= 50个输出维度
    // AnchorBox = 13*13 * 50 = 8450
    // AnchorBox Format:   (|x|y|w|h|confidence|classes_prob| ) * 5
    //                     (|1|1|1|1|1|5|) * 5 = 50

#ifdef HACK_OUTPUT_FROM_NNIE
    imvt_yolo3_init(&layer, NULL, 0, IMVT_YOLO_QUANT_BASE_HISI);
#else
    imvt_yolo3_init(&layer, NULL, 0, IMVT_YOLO_QUANT_BASE_CAFFE);
    //imvt_yolo2_init(&layer, NULL, 0, IMVT_YOLO_QUANT_BASE_CAFFE);
#endif

}

static bool PairCompare(const std::pair<float, int> &lhs, const std::pair<float, int> &rhs)
{
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float> &v, int N)
{
    std::vector<std::pair<float, int>> pairs;
    for (size_t i = 0; i < v.size(); ++i) {
        pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
    }
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i) {
        result.push_back(pairs[i].second);
    }
    return result;
}

/* Return the top N predictions. */
std::vector<YoloBox> Yolo2::Run(const cv::Mat &img, int N)
{
    yolo_boxs_.clear();
    origin_image_size_ = img.size();
    CaffeProcess(img);
    return yolo_boxs_;
}

/* Load the mean file in binaryproto format. */
void Yolo2::SetMean(const string &mean_file)
{
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
            << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float *data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
    * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(caffe_input_image_size_, mean.type(), channel_mean);
}

/* Load the mean file in binaryproto format. */
void Yolo2::SetMean(const string& mean_file, const string& mean_value)
{
    cv::Scalar channel_mean;
    if (!mean_file.empty()) {
        CHECK(mean_value.empty()) <<
                                  "Cannot specify mean_file and mean_value at the same time";
        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

        /* Convert from BlobProto to Blob<float> */
        Blob<float> mean_blob;
        mean_blob.FromProto(blob_proto);
        CHECK_EQ(mean_blob.channels(), num_channels_)
                << "Number of channels of mean file doesn't match input layer.";

        /* The format of the mean file is planar 32-bit float BGR or grayscale. */
        std::vector<cv::Mat> channels;
        float* data = mean_blob.mutable_cpu_data();
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
            channels.push_back(channel);
            data += mean_blob.height() * mean_blob.width();
        }

        /* Merge the separate channels into a single image. */
        cv::Mat mean;
        cv::merge(channels, mean);

        /* Compute the global mean pixel value and create a mean image
        * filled with this value. */
        channel_mean = cv::mean(mean);
        mean_ = cv::Mat(caffe_input_image_size_, mean.type(), channel_mean);
    }
    if (!mean_value.empty()) {
        CHECK(mean_file.empty()) <<
                                 "Cannot specify mean_file and mean_value at the same time";
        stringstream ss(mean_value);
        vector<float> values;
        string item;
        while (getline(ss, item, ',')) {
            float value = std::atof(item.c_str());
            values.push_back(value);
        }
        CHECK(values.size() == 1 || values.size() == num_channels_) <<
                "Specify either 1 mean_value or as many as channels: " << num_channels_;

        std::vector<cv::Mat> channels;
        for (int i = 0; i < num_channels_; ++i) {
            /* Extract an individual channel. */
            cv::Mat channel(caffe_input_image_size_.height, caffe_input_image_size_.width, CV_32FC1,
                            cv::Scalar(values[i]));
            channels.push_back(channel);
        }
        cv::merge(channels, mean_);
    }
}

void Yolo2::CaffeProcess(const cv::Mat &img)
{
    FILE *ptr = NULL;
    (void *)ptr;
    int hacking_output = 0;

    Blob<float> *input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_, caffe_input_image_size_.height, caffe_input_image_size_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    Preprocess(img, &input_channels);

#if HACK_INPUT_FROM_NNIE
    // hack the final input from darknet
    float *input_data = input_layer->mutable_cpu_data();
    //char *hack_input = "yolo2_net_input.bin";
    char *hack_input = "seg0_layer0_output0_func.linear.hex.bgr.float.normalize_1.bin";
    fopen_s(&ptr, hack_input, "rb");
    //fopen_s(&ptr, "street_cars_416x416.bgr", "rb");
    //fopen_s(&ptr, "street_cars_network_input.bin", "rb");
    //fopen_s(&ptr, "street_cars_network_input.bin", "rb");
    printf("loading hack file %s\n", hack_input);
    fread(input_data, 416 * 416 * 3 * 4, 1, ptr);
    fclose(ptr);
#endif

#ifdef HACK_OUTPUT_FROM_NNIE
    hacking_output = 1;
#endif
#ifdef HACK_OUTPUT_FROM_DARKNET
    hacking_output = 1;
#endif
    //Preprocess(img, &input_channels, 0.003921568627451);
    if (hacking_output == 0) {
        net_->Forward();
    }

#ifdef HACK_OUTPUT_FROM_DARKNET
    // hack the final output from darknet
    printf("Forward done\n");
    Blob<float> *output_layer = net_->output_blobs()[0];
    const float *begin = output_layer->cpu_data();
    FILE *ptr;
    fopen_s(&ptr, "nnie_network_output.bin", "rb");
    printf("loading hack file nnie_network_output.bin\n");
    fread((void*)begin, sizeof(float), layer.outputs*layer.batch, ptr);
    fclose(ptr);
#endif

#ifdef HACK_OUTPUT_FROM_NNIE
    printf("HACK_FROM_NNIE\n");
    float *blob0 = (float *)malloc(13 * 13 * 255 * 4);
    float *blob1 = (float *)malloc(26 * 26 * 255 * 4);
    float *blob2 = (float *)malloc(54 * 54 * 255 * 4);

    layer.sub_layer_continue_memory = 0;
    layer.sub_layer[0].w = L1_WIDTH;
    layer.sub_layer[0].h = L1_WIDTH;
    layer.sub_layer[0].output = blob0;
    layer.sub_layer[0].outputs = L1_WIDTH * L1_WIDTH * layer.n * (layer.coords + 1 + layer.classes);
    layer.sub_layer[0].mask[0] = 6;
    layer.sub_layer[0].mask[1] = 7;
    layer.sub_layer[0].mask[2] = 8;

    layer.sub_layer[1].w = L2_WIDTH;
    layer.sub_layer[1].h = L2_WIDTH;
    layer.sub_layer[1].output = blob1;
    layer.sub_layer[1].outputs = L2_WIDTH * L2_WIDTH * layer.n * (layer.coords + 1 + layer.classes);
    layer.sub_layer[1].mask[0] = 3;
    layer.sub_layer[1].mask[1] = 4;
    layer.sub_layer[1].mask[2] = 5;

    layer.sub_layer[2].w = L3_WIDTH;
    layer.sub_layer[2].h = L3_WIDTH;
    layer.sub_layer[2].output = blob2;
    layer.sub_layer[2].outputs = L3_WIDTH * L3_WIDTH * layer.n * (layer.coords + 1 + layer.classes);
    layer.sub_layer[2].mask[0] = 0;
    layer.sub_layer[2].mask[1] = 1;
    layer.sub_layer[2].mask[2] = 2;

    fopen_s(&ptr, "nnie_network_output_0.bin", "rb");
    printf("loading hack file nnie_network_output_0.bin\n");
    fread((void*)layer.sub_layer[0].output, sizeof(float), layer.sub_layer[0].outputs*layer.batch, ptr);
    fclose(ptr);

    fopen_s(&ptr, "nnie_network_output_2.bin", "rb");
    printf("loading hack file nnie_network_output_2.bin\n");
    fread((void*)layer.sub_layer[1].output, sizeof(float), layer.sub_layer[1].outputs*layer.batch, ptr);
    fclose(ptr);

    fopen_s(&ptr, "nnie_network_output_4.bin", "rb");
    printf("loading hack file nnie_network_output_4.bin\n");
    fread((void*)layer.sub_layer[2].output, sizeof(float), layer.sub_layer[2].outputs*layer.batch, ptr);
    fclose(ptr);

    int *p_int0 = (int *)blob0;
    int *p_int1 = (int *)blob1;
    int *p_int2 = (int *)blob2;
    for (int i = 0; i < 10; i++) {
        printf("network 0 output[%d]:%d %f\n", i, p_int0[i], p_int0[i] / 4096.0 );
    }
    for (int i = 0; i < 10; i++) {
        printf("network 1 output[%d]:%d %f\n", i, p_int1[i], p_int1[i] / 4096.0);
    }
    for (int i = 0; i < 10; i++) {
        printf("network 2 output[%d]:%d %f\n", i, p_int2[i], p_int2[i] / 4096.0);
    }
#endif
    PostProcess(img);
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Yolo2::WrapInputLayer(std::vector<cv::Mat> *input_channels)
{
    Blob<float> *input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Yolo2::Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels)
{
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1) {
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4 && num_channels_ == 1) {
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    } else if (img.channels() == 4 && num_channels_ == 3) {
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    } else if (img.channels() == 1 && num_channels_ == 3) {
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    } else {
        sample = img;
    }

    cv::Mat sample_resized;

    if (layer.type == LAYER_YOLO) {
        same_scale_factor = 1;
    } else {
        same_scale_factor = 0;
    }

    if (same_scale_factor) {
        if (sample.size() != caffe_input_image_size_) {
            //cv::Mat img_big(caffe_input_image_size_.width, caffe_input_image_size_.height, CV_32UC3, cv::Scalar(128, 128, 128)); Bug
            cv::Mat img_big(caffe_input_image_size_.width, caffe_input_image_size_.height, CV_8UC3, cv::Scalar(128, 128, 128));
            cv::Mat img_roi;

            cv::Mat img_sacled;
            int new_w = sample.size().width;
            int new_h = sample.size().height;
            if (((float)caffe_input_image_size_.width / new_w) < ((float)caffe_input_image_size_.height / new_h)) {
                new_w = caffe_input_image_size_.width;
                new_h = (sample.size().height * caffe_input_image_size_.width) / sample.size().width;
                cv::resize(sample, img_sacled, cv::Size(new_w, new_h));

                cv::Rect roi(cv::Point(0, (caffe_input_image_size_.height-new_h)/2), cv::Size(new_w, new_h));
                img_roi = img_big(roi);
            } else {
                new_h = caffe_input_image_size_.height;
                new_w = (sample.size().width * caffe_input_image_size_.height) / sample.size().height;
                //printf("2 %d %d\n", new_w, new_h);
                cv::resize(sample, img_sacled, cv::Size(new_w, new_h));
                cv::Rect roi(cv::Point((caffe_input_image_size_.width - new_w)/2, 0), cv::Size(new_w, new_h));
                img_roi = img_big(roi);
            }
            img_sacled.copyTo(img_roi);
            sample_resized = img_big;
            //std::cerr << "roi:" << roi << std::endl;
#if TRACK_ALL_LAYER
            cv::imwrite("img_temp_big.jpg", img_big);
            cv::imwrite("img_temp_sacled.jpg", img_sacled);
            cv::imwrite("img_temp_roi.jpg", img_roi);
#endif
        } else {
            //std::cerr << "no need to resize image" << std::endl;
            sample_resized = sample;
        }
    } else {
        if (sample.size() != caffe_input_image_size_) {
            cv::resize(sample, sample_resized, caffe_input_image_size_);
            //std::cerr << "resize image to 416x416 (directly)" << std::endl;
        } else {
            //std::cerr << "no need to resize image" << std::endl;
            sample_resized = sample;
        }
    }
    //DEBUG
#if TRACK_ALL_LAYER
    cv::imwrite("img_temp_sample_resized.jpg", sample_resized);
#endif

    cv::Mat sample_float;
    if (num_channels_ == 3) {
        sample_resized.convertTo(sample_float, CV_32FC3);
    } else {
        sample_resized.convertTo(sample_float, CV_32FC1);
        std::cout << "[Error] caffe must be 3 channel. see prototxt!" << std::endl;
    }

#if 1
    cv::Mat sample_normalized;
    //cv::subtract(sample_float, mean_, sample_normalized);
    cv::normalize(sample_float, sample_normalized, 0, 1, CV_MINMAX, CV_32FC3);

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    //cv::Mat xxximg = cv::imread("input.png");
    //cv::split(xxximg, *input_channels);
#else
    cv::split(sample_float, *input_channels);
#endif

    CHECK(reinterpret_cast<float *>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}

void Yolo2::Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels, double normalize_value)
{
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1) {
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4 && num_channels_ == 1) {
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    } else if (img.channels() == 4 && num_channels_ == 3) {
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    } else if (img.channels() == 1 && num_channels_ == 3) {
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    } else {
        sample = img;
    }

    cv::Mat sample_resized;
    if (sample.size() != caffe_input_image_size_) {
        cv::resize(sample, sample_resized, caffe_input_image_size_);
    } else {
        sample_resized = sample;
    }

    cv::Mat sample_float;
    if (num_channels_ == 3) {
        std::cout << "normalize_value:" << normalize_value << std::endl;
        sample_resized.convertTo(sample_float, CV_32FC3, normalize_value);
    } else {
        sample_resized.convertTo(sample_float, CV_32FC1, normalize_value);
        std::cout << "[Error] caffe must be 3 channel. see prototxt!" << std::endl;
    }

#if 1
    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);
    //cv::normalize(sample_float, sample_normalized, 0, 1, CV_MINMAX, CV_32FC3);

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);
#else
    cv::split(sample_float, *input_channels);
#endif

    CHECK(reinterpret_cast<float *>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}

void Yolo2::SelectYoloBox(detection *dets, int nbox)
{
    int i, j;
    YoloBox ybox;
    int img_w = origin_image_size_.width;
    int img_h = origin_image_size_.height;

    // caffe image size vs. origin image size
    //float x_scale = (float)origin_image_size_.width / (float)caffe_input_image_size_.width;
    //float y_scale = (float)origin_image_size_.height / (float)caffe_input_image_size_.height;

    //printf("img_w:%d img_h:%d\n", img_w, img_h);
    //printf("num:%d, confidence_threshold_:%f classes:%d\n", nbox, confidence_threshold_, classes_);

    for(i = 0; i < nbox; ++i) {
        //char labelstr[4096] = {0};
        int class_ = -1;
        float prob = 0.0f;
        int class_idx;
        for(j = 0; j < classes_; ++j) {
            if (dets[i].prob[j] > confidence_threshold_) {
                if (class_ < 0) {
                    //strcat(labelstr, names[j]);
                    class_ = j;
                } else {
                    //strcat(labelstr, ", ");
                    //strcat(labelstr, names[j]);
                }
                //printf("%s: %.0f%%\n", names[j], dets[i].prob[j]*100);
                //printf("classes idx:%d prob:%f\n", j, dets[i].prob[j]);
                prob = dets[i].prob[j];
                class_idx = j;
            }
        }

        if(class_ >= 0) {
            box b = dets[i].bbox;
            //printf("yolov2 anchorbox[%d]:%f %f %f %f class:%d prob:%f\n", i, b.x, b.y, b.w, b.h, class_, prob);

            int left   = (b.x-b.w/2.)*img_w;
            int right  = (b.x+b.w/2.)*img_w;
            int top    = (b.y-b.h/2.)*img_h;
            int bottom = (b.y+b.h/2.)*img_h;

            if (left < 0) {
                left = 0;
            }
            if (right > img_w-1) {
                right = img_w-1;
            }
            if (top < 0) {
                top = 0;
            }
            if (bottom > img_h-1) {
                bottom = img_h-1;
            }
            ybox.left = left;
            ybox.top = top;
            ybox.right = right;
            ybox.bottom = bottom;
            ybox.confidence = prob;
            ybox.class_idx = class_idx;
            ybox.class_name = (char *)class_names_[class_idx];
            this->yolo_boxs_.push_back(ybox);

            if (dets[i].mask) {

            }
        }
    }
}

/* parse last layer output */
void Yolo2::PostProcess(const cv::Mat &img)
{
    int j;
    Blob<float> *output_layer;
    const float *begin;
    char name[100];
    FILE *ptr;
    int outputs;
    int sub_layer_idx;

    if (net_->num_outputs() > 3) {
        printf("not support network\n");
        return;
    }
    //printf("net_->num_outputs():%d\n", net_->num_outputs());

    if (net_->num_outputs() > 1) {
        // not linear network output (not flatten, not concat)
        layer.sub_layer_continue_memory = 0;
        layer.sub_layer_num = net_->num_outputs();
        //printf("layer.sub_layer_num:%d\n", layer.sub_layer_num);
    }

    //fill in network output pointer:
    // yolov3 have 3 output
    // yolov3 mini have 2 output
    // yolov1 yolov2 have 1 output
    for (int j = 0; j < net_->num_outputs(); j++) {
        output_layer = net_->output_blobs()[j];
        begin = output_layer->cpu_data();
        //begin = output_layer->gpu_data();
        if (output_layer->width() == 13) {
            sub_layer_idx = 0;
        } else if (output_layer->width() == 26) {
            sub_layer_idx = 1;
        } else if (output_layer->width() == 52) {
            sub_layer_idx = 2;
        } else {
            sub_layer_idx = 0;
        }
        layer.sub_layer[sub_layer_idx].output = (float *)begin;
        layer.output = (float*)begin; //always assign the last output

#if TRACK_ALL_LAYER
        printf("blob[%d] width:%d height:%d channel:%d \n", j, output_layer->width(), output_layer->height(), 0/*output_layer*/);
        {
            FILE *ptr;
            ptr = fopen("network_output.bin", "wb");
            fwrite((void*)begin, 1, layer.outputs*layer.batch * sizeof(float), ptr);
            fclose(ptr);
        }
#endif

#if TRACK_ALL_LAYER
        //Hack,dump
        for (int i = 0; i < 10; i++) {
            printf("network output[%d] idx:[%d]:%f\n", j, i, begin[i]);
        }
        sprintf(name, "caffe_network_output_%s_%d_width_%d.bin", net_->name().c_str(), j, output_layer->width());
        //fopen_s(&ptr, name, "wb");
        ptr = fopen(name, "wb");
        fwrite((void*)begin, sizeof(float), layer.sub_layer[j].outputs, ptr);
        fclose(ptr);
#endif
    }
    float hier_thresh = 0.5f;
    int relative = 1;

    int nboxes = 0;
    detection *dets;
#if 0
    // Note:
    //   1. The Darknet author change the network a lot, and not sure the which version need resize, which don't.
    //      some need region_box correction, some don't
    //   2. need no letterbox is use, maybe correct_region_boxes can be skip for optimize.
    // Test it yourself
    dets = imvt_yolo3_get_detection(&layer, origin_image_size_.width, origin_image_size_.height, confidence_threshold_, hier_thresh, 0, relative, &nboxes);
    dets = imvt_yolo3_get_detection(&layer,  layer.net_w, layer.net_h, confidence_threshold_, hier_thresh, 0, relative, &nboxes);
#endif

    if (layer.type == LAYER_YOLO) {
        dets = imvt_yolo3_get_detection(&layer, origin_image_size_.width, origin_image_size_.height, confidence_threshold_, hier_thresh, 0, relative, &nboxes);
    } else {
        dets = imvt_yolo3_get_detection(&layer, layer.net_w, layer.net_h, confidence_threshold_, hier_thresh, 0, relative, &nboxes);
    }

    if (nms_threshold_ > 0.0f) {
        imvt_yolo3_do_nms_sort(dets, nboxes, layer.classes, nms_threshold_);
    }
    SelectYoloBox(dets, nboxes);
    imvt_yolo3_free_detection(dets, nboxes);
}
