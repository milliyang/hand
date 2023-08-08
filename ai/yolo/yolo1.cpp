#include "stdafx.h"
#include "yolo1.h"
#include "box.h"

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

const static char *yolo1_class_name[] = {
    //"background  ",
    "plane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
};

Yolo1::Yolo1(const string &model_file, const string &trained_caffemodel, const string &mean_file, const string &label_file)
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

    Blob<float> *output_layer = net_->output_blobs()[0];
    std::cout << "output_layer->channels:" << output_layer->channels() << std::endl;
    // Blob<float> *output_layer = net_->output_blobs()[0];
    // CHECK_EQ(labels_.size(), output_layer->channels())
    // 	<< "Number of labels is different from the output layer dimension.";

    //layer output information
    confidence_threhold_ = 0.15;
    grid_ = 7;
    bbox_each_grid = 2;
    classes_ = 20;
    total_bbox_ = grid_ * grid_ * bbox_each_grid; //7 * 7 * 2;
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
std::vector<YoloBox> Yolo1::Run(const cv::Mat &img, int N)
{
    yolo_boxs_.clear();
    //origin_image_width_ = img.size().width;
    //image.size().height
    origin_image_size_ = img.size();
    CaffeProcess(img);
    return yolo_boxs_;
}

/* Load the mean file in binaryproto format. */
void Yolo1::SetMean(const string &mean_file)
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

void Yolo1::CaffeProcess(const cv::Mat &img)
{
    Blob<float> *input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_, caffe_input_image_size_.height, caffe_input_image_size_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    Preprocess(img, &input_channels);
    net_->Forward();
    PostProcess(img);
}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void Yolo1::WrapInputLayer(std::vector<cv::Mat> *input_channels)
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

void Yolo1::Preprocess(const cv::Mat &img, std::vector<cv::Mat> *input_channels)
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
#else
    cv::split(sample_float, *input_channels);
#endif

    CHECK(reinterpret_cast<float *>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}

typedef struct {
    int index;
    int classes;
    float **probs;
} sortable_bbox;

static float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

static float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) {
        return 0;
    }
    float area = w * h;
    return area;
}

static float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}

static float box_iou(box a, box b)
{
    return box_intersection(a, b) / box_union(a, b);
}

static void get_detection_boxes(const float *addr, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int grid = 7; // N*N grids
    int bbox = 2; // N bbox for each grid
    int classes = 20;
    int is_sqrt = 1;

    int i, j, n;

    const float *predictions = addr;
    //int per_cell = 5*num+classes;
    for (i = 0; i < grid * grid; ++i) {
        int row = i / grid;
        int col = i % grid;
        for (n = 0; n < bbox; ++n) {
            int index = i * bbox + n;
            int p_index = grid * grid * classes + i * bbox + n;
            float scale = predictions[p_index];
            int box_index = grid * grid * (classes + bbox) + (i * bbox + n) * 4;
            boxes[index].x = (predictions[box_index + 0] + col) / grid * w;
            boxes[index].y = (predictions[box_index + 1] + row) / grid * h;
            boxes[index].w = pow(predictions[box_index + 2], (is_sqrt ? 2 : 1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (is_sqrt ? 2 : 1)) * h;
            for (j = 0; j < classes; ++j) {
                int class_index = i * classes;
                float prob = scale * predictions[class_index + j];
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
            if (only_objectness) {
                probs[index][0] = scale;
            }
        }
    }
}

static void print_yolo_detections(box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for (i = 0; i < total; ++i) {
        float xmin = boxes[i].x - boxes[i].w / 2.;
        float xmax = boxes[i].x + boxes[i].w / 2.;
        float ymin = boxes[i].y - boxes[i].h / 2.;
        float ymax = boxes[i].y + boxes[i].h / 2.;

        if (xmin < 0) {
            xmin = 0;
        }
        if (ymin < 0) {
            ymin = 0;
        }
        if (xmax > w) {
            xmax = w;
        }
        if (ymax > h) {
            ymax = h;
        }

        for (j = 0; j < classes; ++j) {
            if (probs[i][j]) {
                fprintf(stdout, "[%s] %f %f %f %f %f   class_idx:%d\n", yolo1_class_name[j], probs[i][j], xmin, ymin, xmax, ymax, j);
            }
        }
    }
}

static int nms_comparator(const void *pa, const void *pb)
{
    sortable_bbox a = *(sortable_bbox *)pa;
    sortable_bbox b = *(sortable_bbox *)pb;
    float diff = a.probs[a.index][b.classes] - b.probs[b.index][b.classes];
    if (diff < 0) {
        return 1;
    } else if (diff > 0) {
        return -1;
    }
    return 0;
}

static void do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    sortable_bbox *s = (sortable_bbox*)calloc(total, sizeof(sortable_bbox));

    for (i = 0; i < total; ++i) {
        s[i].index = i;
        s[i].classes = 0;
        s[i].probs = probs;
    }

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            s[i].classes = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator);
        for (i = 0; i < total; ++i) {
            if (probs[s[i].index][k] == 0) {
                continue;
            }
            box a = boxes[s[i].index];
            for (j = i + 1; j < total; ++j) {
                box b = boxes[s[j].index];
                if (box_iou(a, b) > thresh) {
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    free(s);
}

void Yolo1::SelectYoloBox(box *boxes, float **probs)
{
    YoloBox ybox;
    int w = caffe_input_image_size_.width;
    int h = caffe_input_image_size_.height;

    // caffe image size vs. origin image size
    float x_scale = (float)origin_image_size_.width / (float)caffe_input_image_size_.width;
    float y_scale = (float)origin_image_size_.height / (float)caffe_input_image_size_.height;

    int i, j;
    for (i = 0; i < this->total_bbox_; ++i) {
        float xmin = boxes[i].x - boxes[i].w / 2.;
        float xmax = boxes[i].x + boxes[i].w / 2.;
        float ymin = boxes[i].y - boxes[i].h / 2.;
        float ymax = boxes[i].y + boxes[i].h / 2.;

        if (xmin < 0) {
            xmin = 0;
        }
        if (ymin < 0) {
            ymin = 0;
        }
        if (xmax > w) {
            xmax = w;
        }
        if (ymax > h) {
            ymax = h;
        }

        for (j = 0; j < this->classes_; ++j) {
            if (probs[i][j]) {
                //fprintf(stdout, "[%s] %f %f %f %f %f   class_idx:%d\n", yolo1_class_name[j], probs[i][j], xmin, ymin, xmax, ymax, j);
                ybox.left = xmin * x_scale;
                ybox.top = ymin * y_scale;
                ybox.right = xmax * x_scale;
                ybox.bottom = ymax * y_scale;
                ybox.confidence = probs[i][j];
                ybox.class_idx = j;
                ybox.class_name = (char*) yolo1_class_name[j];
                this->yolo_boxs_.push_back(ybox);
            }
        }
    }
}

/* parse last layer output */
void Yolo1::PostProcess(const cv::Mat &img)
{
    int nms = 1;
    float iou_thresh = 0.5f;
    int j;
    /* Copy the output layer to a std::vector */
    Blob<float> *output_layer = net_->output_blobs()[0];
    const float *begin = output_layer->cpu_data();

    box *boxes = (box *) calloc(total_bbox_, sizeof(box));
    float **probs = (float **) calloc(total_bbox_, sizeof(float *));
    for (j = 0; j < total_bbox_; ++j) {
        probs[j] = (float*) calloc(classes_, sizeof(float *));
    }

    get_detection_boxes(begin, caffe_input_image_size_.width, caffe_input_image_size_.height, confidence_threhold_, probs, boxes, 0);

    if (nms) {
        do_nms_sort(boxes, probs, total_bbox_, classes_, iou_thresh);
    }
    //print_yolo_detections(boxes, probs, total_bbox_, classes_, caffe_input_image_size_.width, caffe_input_image_size_.height);
    SelectYoloBox(boxes, probs);

    // for (auto ybox : yolo_boxs_) {
    // 	std::cout << yolo1_class_name[ybox.class_idx] << ybox.confidence << " " << ybox.left << " " << ybox.top << " " << ybox.right << " " << ybox.bottom << std::endl;
    // }
}
