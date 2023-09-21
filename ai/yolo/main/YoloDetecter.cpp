#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include "yolo1.h"
#include "yolo2.h"
#include "yolo2ab.h"
#include "proxy_reader.h"
#include "imvt_cv_tracking.h"

#include <iostream>
#include <fstream>

#ifdef CONFIG_SPDLOG
#define LOG_TAG "main"
#include "log.h"
#endif

using namespace std;

// ./detect yolov2.prototxt yolov2.caffemodel car.mov
// ./detect yolov2.prototxt yolov2.caffemodel person.jpg
// ./detect yolov2.prototxt yolov2.caffemodel tcp://192.168.9.6:5556
// ./detect darknet2caffe/imvt20_yolo2.cut.prototxt darknet2caffe/imvt20_yolo2.caffemodel ~/coco/coco_val2017_filelists.txt

// Batch generate xxxx.jpg.yolo.txt detection result for mAP
//  ./detect darknet2caffe/imvt20_yolo2.cut.prototxt darknet2caffe/imvt20_yolo2.caffemodel ~/coco/coco_val2017_filelists.txt 1
void simple_draw_yolo_result(cv::Mat frame, int seq, std::vector<YoloBox> &yolo_boxs)
{
    cv::Rect_<float> box;
    cv::Point pt_text;
    for (auto ybox : yolo_boxs) {
        box.x = ybox.left;
        box.y = ybox.top;
        box.width = ybox.right - ybox.left;
        box.height = ybox.bottom - ybox.top;

        // then put the text itself
        pt_text.x = box.x;
        pt_text.y = box.y > 2 ? box.y - 5 : 0;
        stringstream ss;
        //ss.precision(2);
        ss << "(" << ybox.class_idx << ") " <<  Yolo2::id_to_name(ybox.class_idx) << " |" << (int)(ybox.confidence*100);

        cv::rectangle(frame, box, cv::Scalar(255, 0, 0));
        putText(frame, ss.str(), pt_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 250, 0), 1, 8);
    }
}

std::vector<YoloBox> selected_class_with_thresh(std::vector<YoloBox> &yolo_boxs)
{
    std::vector<YoloBox> selected;
    for (int i = 0; i < yolo_boxs.size(); i++) {
        //SORT_CLS_HUMAN
        //SORT_CLS_FACE
        //SORT_CLS_HAND
        //thresh
        if (yolo_boxs[i].confidence >= 0.01f) {
            selected.push_back(yolo_boxs[i]);
        }
    }
    return selected;
}

void write_yolobox_to_file(std::string filename, std::vector<YoloBox> &yolo_boxs)
{
    std::fstream fs;
    fs = fstream(filename, std::ios::out|std::ios::trunc);

    for (auto ybox: yolo_boxs) {
        float cx = (ybox.right + ybox.left)/2.0f;
        float cy = (ybox.bottom + ybox.top)/2.0f;
        float w = (ybox.right - ybox.left);
        float h = (ybox.bottom - ybox.top);
        float prob = ybox.confidence;

        cx /= 416.0f;
        cy /= 416.0f;
        w /= 416.0f;
        h /= 416.0f;

        stringstream ss;
        ss.precision(2);
        ss << ybox.class_idx << " " << prob << " " << cx << " " << cy << " " << w << " " << h << "\n";
        std:string info = ss.str();
        fs.write(info.c_str(), info.size());
    }
    fs.close();
}

int main(int argc, char **argv)
{
    int ret;
    int detect = 1;
    int source = 0;
    int show_once = 1;
    int pause = 0;
    int pause_frame_to_process = 0; //
    int frame_seq = 0;
    int batch = 0;

    if (argc < 4) {
        std::cerr << "Usage: "
                  << argv[0] << " deploy.prototxt network.caffemodel xxx.mov" << std::endl
                  << argv[0] << " deploy.prototxt network.caffemodel /dev/video0" << std::endl
                  << argv[0] << " deploy.prototxt network.caffemodel filelist.txt" << std::endl;
        return 1;
    }
    if (argc == 5) {
        batch = atoi(argv[4]);
    }

    ::google::InitGoogleLogging(argv[0]);

    string model_file = argv[1];
    string trained_file = argv[2];
    string mov_file = argv[3];
    Yolo2 yolo(model_file, trained_file, model_file, model_file);

    imvt_cv_tracking_init();

    int input_tracking_num = -1;

    ProxyReader reader;

    if (!reader.open(mov_file)) {
        return 0;
    }
    std::vector<YoloBox> yolo_boxs;

    if (reader.isFilelist()) {
        pause = 1;
    }



    while (1) {
        cv::Mat frame;
        cv::Rect2d cv_box;
        std::vector<YoloBox> selected;
        std::string filename;

        if (batch) {
            filename = reader.peekFile();
        }

        if (!batch && pause && pause_frame_to_process <= 0 && frame_seq > 0) {
            //do nothing
        } else {
            reader.read(frame);

            // If the frame is empty, break immediately
            if (frame.empty()) {
                break;
            }

            cv::resize(frame, frame, cv::Size(416, 416));

            if (show_once && !batch) {
                show_once = 0;
                cv::imshow("YoloTracker", frame);
            }

            if (!reader.isStream() || detect) {
                double timer = (double)cv::getTickCount();
                std::vector<YoloBox> yolo_boxs = yolo.Run(frame);
                selected = selected_class_with_thresh(yolo_boxs);

                if (!batch) {
                    simple_draw_yolo_result(frame, frame_seq, selected);

                    // Calculate Frames per second (FPS)
                    float fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);
                    stringstream ss;
                    ss.setf(std::ios::fixed);
                    ss.precision(2);
                    ss << "FPS:" << fps << " SEQ:" << frame_seq;
                    putText(frame, ss.str(), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, 8);
                }
            }

            if (!batch) {
                cv::imshow("YoloTracker",  frame);
            }

            if (pause_frame_to_process > 0) { pause_frame_to_process--;}
            frame_seq++;
        }
        if (batch) {
            // /home/leo/coco/images/val2017/000000532481.jpg
            // /home/leo/coco/images/val2017/000000532481.jpg.yolo.txt
            std::string label = filename + ".yolo.txt";
            write_yolobox_to_file(label, selected);
            continue;
        }

        // Press  ESC on keyboard to exit
        char c;
        if (reader.isStream()) {
            c = (char)cv::waitKey(15);
        } else {
            c = (char)cv::waitKey(15 * 10000);
        }
        //if (c >= 0) { std::cout << "keyboard:" << (int)c << std::endl; }

        if (c == 112 /* p */) {
            pause = !pause;
            pause_frame_to_process = 0;
        } else if (c == 110 /* n */) {
            pause = 1;
            pause_frame_to_process++;
        } else if (c >= 48 && c <= 57 /* 0~9 */) {
            int num = c - 48;
            if (input_tracking_num < 0) {
                input_tracking_num = num;
            } else {
                input_tracking_num = input_tracking_num * 10 + num;
            }
        } else if (c == 27 /* Escape */ || c == 113 /* 'q' */) {
            break;
        } else if (c == 116 /* 't' */) {
            detect = !detect;
        } else if (c == 108 /* 'l' */) {
            reader.resetReadIndex(-1);  //last file
            pause_frame_to_process++;
        } else if (c == 115 /* 's' */) {
            cv::Rect2d bbox = selectROI("YoloTracker", frame, false);
            imvt_cv_tracking_enable(1, bbox.x, bbox.y, bbox.width, bbox.height);
            detect = 1; //start tracking
        } else {
            //
        }
    }

    reader.close();

    // Closes all the frames
    cv::destroyAllWindows();
    imvt_cv_tracking_deinit();

    return 0;
}
