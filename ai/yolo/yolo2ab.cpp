//#include "stdafx.h"
#include "yolo2ab.h"
#include "box.h"
#include "imvt_yolo2x.h"

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

const static char *yolo2_class_name[] = {
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

Yolo2AB::Yolo2AB(const string &model_file, const string &trained_caffemodel, const string &mean_file, const string &label_file)
{
    Caffe::set_mode(YOLO_CAFFE_MODE);

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_caffemodel);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float> *input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
    caffe_input_image_size_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    //SetMean(mean_file);
    SetMean("", "0,0,0");

    Blob<float> *output_layer = net_->output_blobs()[0];
    std::cout << "output_layer->channels:" << output_layer->channels() << std::endl;
    // Blob<float> *output_layer = net_->output_blobs()[0];
    // CHECK_EQ(labels_.size(), output_layer->channels())
    // 	<< "Number of labels is different from the output layer dimension.";

    layer.type = LAYER_REGION;
    layer.n = 5;
    layer.w = layer.h = 13;
    layer.batch = 1;
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
    layer.classes = 80;
    layer.classfix = 1;
    layer.coords = 4;
    layer.background = 0;
    layer.net_w = 416;
    layer.net_h = 416;
    layer.outputs = layer.w * layer.h * layer.n * (layer.coords + 1 + layer.classes);
    printf("layer.outputs:%d\n", layer.outputs);

    //layer output information
    same_scale_factor = 0;
    confidence_threshold_ = 0.45;
    nms_threshold_ = 0.45;
    grid_ = layer.w;
    bbox_each_grid = layer.n;
    classes_ = layer.classes;
    total_bbox_ = grid_ * grid_ * bbox_each_grid; //13 * 13 * 5; anchor box
    //channel = ( 5 * (5+80) = 425)
    biases_[0] = layer.biases[0];
    biases_[1] = layer.biases[1];
    biases_[2] = layer.biases[2];
    biases_[3] = layer.biases[3];
    biases_[4] = layer.biases[4];
    biases_[5] = layer.biases[5];
    biases_[6] = layer.biases[6];
    biases_[7] = layer.biases[7];
    biases_[8] = layer.biases[8];
    biases_[9] = layer.biases[9];
    class_names_ = yolo2_class_name;

    //eg.
    //每个box包含5个坐标值和20个类别，所以总共是5 * （5+20）= 125个输出维度
    //hisi:
    //每个box包含5个坐标值和5个类别，所以总共是5 * （5+5）= 50个输出维度
    // AnchorBox = 13*13 * 50 = 8450
    // AnchorBox Format:   (|x|y|w|h|confidence|classes_prob| ) * 5
    //                     (|1|1|1|1|1|5|) * 5 = 50
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
std::vector<YoloBox> Yolo2AB::Run(const cv::Mat &img, int N)
{
    yolo_boxs_.clear();
    origin_image_size_ = img.size();
    CaffeProcess(img);
    return yolo_boxs_;
}

/* Load the mean file in binaryproto format. */
void Yolo2AB::SetMean(const string &mean_file)
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
void Yolo2AB::SetMean(const string& mean_file, const string& mean_value)
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

void Yolo2AB::CaffeProcess(const cv::Mat &img)
{
    Blob<float> *input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_, caffe_input_image_size_.height, caffe_input_image_size_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    Preprocess(img, &input_channels);

#if 0
    // hack the final input from darknet
    float *input_data = input_layer->mutable_cpu_data();
    FILE *ptr;
    fopen_s(&ptr, "network_input.bin", "rb");  // r for read, b for binary
    fread(input_data, 416*416*3*4, 1, ptr); // read 10 bytes to our buffer
#endif

    //Preprocess(img, &input_channels, 0.003921568627451);
    net_->Forward();

    PostProcess(img);
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Yolo2AB::WrapInputLayer(std::vector<cv::Mat> *input_channels)
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

void Yolo2AB::Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels)
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
                printf("2 %d %d\n", new_w, new_h);

                cv::resize(sample, img_sacled, cv::Size(new_w, new_h));
                cv::Rect roi(cv::Point((caffe_input_image_size_.width - new_w)/2, 0), cv::Size(new_w, new_h));
                img_roi = img_big(roi);
            }
            img_sacled.copyTo(img_roi);
            sample_resized = img_big;
            //std::cerr << "roi:" << roi << std::endl;
            //cv::imwrite("img_big.jpg", img_big);
            //cv::imwrite("img_sacled.jpg", img_sacled);
            //cv::imwrite("img_roi.jpg", img_roi);
        } else {
            sample_resized = sample;
        }
    } else {
        if (sample.size() != caffe_input_image_size_) {
            cv::resize(sample, sample_resized, caffe_input_image_size_);
        } else {
            sample_resized = sample;
        }
    }

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

void Yolo2AB::Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels, double normalize_value)
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

void Yolo2AB::SelectYoloBox(detection *dets, int nbox)
{
    int i, j;
    YoloBox ybox;
    int img_w = origin_image_size_.width;
    int img_h = origin_image_size_.height;

    printf("num:%d, confidence_threshold_:%f classes:%d\n", nbox, confidence_threshold_, classes_);

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
            ybox.class_name = (char*) yolo2_class_name[class_idx];
            this->yolo_boxs_.push_back(ybox);

            if (dets[i].mask) {

            }
        }
    }
}

struct bbox_t {
    unsigned int x, y, w, h;	// (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;					// confidence - probability that the object was found correctly
    unsigned int obj_id;		// class of object - from range [0, classes-1]
    unsigned int track_id;		// tracking id for video (0 - untracked, 1 - inf - tracked object)
};

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

/* parse last layer output */
void Yolo2AB::PostProcess(const cv::Mat &img)
{
    int j;
    /* Copy the output layer to a std::vector */
    Blob<float> *output_layer = net_->output_blobs()[0];
    const float *begin = output_layer->cpu_data();
    layer.output = (float*)begin;

    printf("network_output size:%d\n", layer.outputs*layer.batch * sizeof(float));

#if 0
    FILE *ptr;
    fopen_s(&ptr, "network_output.bin", "wb");
    fwrite((void*)begin, 1, layer.outputs*layer.batch * sizeof(float), ptr);
    fclose(ptr);
#endif

#if 0
    // hack the final output from darknet
    FILE *ptr2;
    errno_t err;
    err = fopen_s(&ptr2, "darknet_network_output.bin", "rb");
    if (err != 0) {
        printf("The file 'darknet_network_output.bin' was not opened\n");
    }
    fread((void*)begin, sizeof(float), layer.outputs*layer.batch, ptr2);
    fclose(ptr2);
#endif

    layer.output = (float*)begin;
    float hier_thresh = 0.5f;
    int relative = 1;

    int nboxes;
    detection *dets;
    dets = imvt_yolo2x_get_detection(&layer, origin_image_size_.width, origin_image_size_.height, confidence_threshold_, hier_thresh, 0, relative, &nboxes, 0);
    if (nms_threshold_ > 0.0f) {
        imvt_yolo2x_do_nms_sort(dets, nboxes, layer.classes, nms_threshold_);
    }
    SelectYoloBox(dets, nboxes);
    imvt_yolo2x_free_detection(dets, nboxes);
}
