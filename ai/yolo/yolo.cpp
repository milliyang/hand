//#include "stdafx.h"

#include "yolo1.h"
#include "yolo2.h"

using namespace caffe;
using std::string;

#if 0
int main(int argc, char **argv)
{
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " deploy.prototxt network.caffemodel"
                  << " mean.binaryproto labels.txt img.jpg" << std::endl;
        return 1;
    }

    //::google::InitGoogleLogging(argv[0]);

    string model_file = argv[1];
    string trained_file = argv[2];
    string mean_file = argv[3];
    string label_file = argv[4];
    Yolo1 yolo1(model_file, trained_file, mean_file, label_file);

    string file = argv[5];
    std::cout << "---------- Prediction for " << file << " ----------" << std::endl;

    cv::Mat img = cv::imread(file, -1);
    //CHECK(!img.empty()) << "Unable to decode image " << file;
    std::vector<YoloBox> yolo_boxs = yolo1.Run(img);

    for (auto ybox : yolo_boxs) {
        std::cout << ybox.class_name << ybox.confidence << " " << ybox.left << " " << ybox.top << " " << ybox.right << " " << ybox.bottom << std::endl;
    }

    // ./Yolo.exe yolo_small_deploy.prototxt yolo_small.caffemodel mean.binaryproto labels.txt person.jpg
}

#endif
