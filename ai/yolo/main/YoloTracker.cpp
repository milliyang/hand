#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include "yolo1.h"
#include "yolo2.h"
#include "yolo2ab.h"
#include "proxy_reader.h"
#include "imvt_cv_tracking.h"

#include "foo_tracker.h"
#include "moving_object.h"

#include "CheapSort.h"
#include "SimilarTracker.h"

#ifdef CONFIG_SPDLOG
#define LOG_TAG "main"
#include "log.h"
#endif

using namespace std;

// ./yolo yolo_small_deploy.prototxt yolo_small.caffemodel yolo.mov
// ./yolo yolo2_hisi/yolo2.prototxt yolo2_hisi/yolo2.caffemodel yolo.mov
// ./yolo yolov2.prototxt yolov2.caffemodel car.mov
// ./yolo yolov2.prototxt yolov2.caffemodel person.jpg
// ./yolo fake.prototxt fake.caffemodel person.jpg

//  from: 192.168.9.6
// ./yolo yolov2.prototxt yolov2.caffemodel tcp://192.168.9.6:5556
// ./yolo yolov2.prototxt yolov2.caffemodel tcp://192.168.9.7:5556

// ./yolo yolov3.mini.prototxt yolov3.mini.caffemodel street_cars.jpg

void runFooTracker(FooTracker &fooTracker, cv::Mat &frame, std::vector<YoloBox> &yolo_boxs)
{
    std::vector<Info> infos;
    Info info;
    info.id = -1;

    for (auto ybox : yolo_boxs) {
        if (ybox.class_idx != 0) {
            continue;
        }

        info.id = -1;
        info.rect.x = ybox.left;
        info.rect.y = ybox.top;
        info.rect.width = ybox.right - ybox.left;
        info.rect.height = ybox.bottom - ybox.top;
        infos.push_back(info);
    }
    fooTracker.updateYoloResult(frame, infos);
}

void drawFooTracker(FooTracker &fooTracker, cv::Mat &frame)
{
    //draw fooTracker result
    std::vector<Info> infos = fooTracker.getAllResult();
    if (infos.begin() == infos.end()) {
        std::stringstream info;
        info << "FooTracking NG";
        putText(frame, info.str(), cv::Point(5, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, 8);
    } else {
        string label;
        for (auto it = infos.begin(); it != infos.end(); it++) {
            if (it == infos.begin()) {
                cv::rectangle(frame, (*it).rect, cv::Scalar(0, 250, 0), 2);
            } else {
                cv::rectangle(frame, (*it).rect, cv::Scalar(0, 250, 0), 1);
            }
            cv::Point pos;
            pos.x = (*it).rect.x;
            pos.y = (*it).rect.y < 12 ? 12 : (*it).rect.y;
            std::stringstream ss;
            ss << "Foo:" << (*it).id;
            putText(frame, ss.str(), pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 250, 0), 1, 8);
        }
    }
}

void drawYoloResult(cv::Mat &frame, int seq, std::vector<YoloBox> &yolo_boxs)
{
    if (yolo_boxs.size() <= 0) {
        std::cout << "not found! frame_seq:" << seq << std::endl;
    } else {
        for (auto ybox : yolo_boxs) {
            stringstream ss;
            ss.precision(2);
            ss << "("<< ybox.class_idx << ")" << Yolo2::id_to_name(ybox.class_idx) << " P|" << ybox.confidence;
            //std::cout << "Yolo :" << Yolo2::id_to_name(ybox.class_idx) << " idx:" << ybox.class_idx << " confidence:" << ybox.confidence << " " << ybox.left << " " << ybox.top << " " << ybox.right << " " << ybox.bottom << " frame_seq:" << seq << std::endl;
            //rectangle(Mat& img, Rect rec, const Scalar& color, int thickness=1, int lineType=8, int shift=0 )
            cv::Point pt_text;
            pt_text.x = ybox.left;
            pt_text.y = ybox.top-2;
            putText(frame, ss.str(), pt_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 250, 250), 1, 8);

            cv::Rect rect(ybox.left, ybox.top, (ybox.right - ybox.left), (ybox.bottom - ybox.top));
            cv::rectangle(frame, rect, cv::Scalar(250, 0, 0));
        }
    }
}

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
        ss.precision(2);
        ss << "(" << -1 << ") " <<  Yolo2::id_to_name(ybox.class_idx) << " P|" << ybox.confidence;

        cv::rectangle(frame, box, cv::Scalar(255, 0, 0));
        putText(frame, ss.str(), pt_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 250, 0), 1, 8);
    }
}

void draw_tracking_boxes_with_style(cv::Mat &frame, const std::vector<TrackingBox> &tboxes, int thickness=1, int lineType = cv::LINE_8)
{
    #define CC_HUMAN_YELLOW     cv::Scalar(0, 250, 250)
    #define CC_FACE_GREEN       cv::Scalar(0, 250,   0)
    #define CC_HAND_BLUE        cv::Scalar(250, 250, 0)

    cv::Point pt_text;
    for (auto tbox : tboxes) {
        pt_text.x = tbox.box.x;
        pt_text.y = tbox.box.y > 2 ? tbox.box.y - 5 : 0;
        stringstream ss;
        ss.precision(2);

        //ss << "(" << tbox.id << ")" << Yolo2::id_to_name(tbox.class_idx) << " P|" << tbox.confidence;
        ss << tbox.id << "-" << (int)(tbox.confidence * 100);

        int thick = thickness;
        if (tbox.tracking) {
            thick = 4;
        }
        if (tbox.class_idx == 0) {
            pt_text.y = pt_text.y+tbox.box.height-10;
            pt_text.y = std::min(pt_text.y, 416);
            putText(frame, ss.str(), pt_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, CC_HUMAN_YELLOW, 1, 8);
            cv::rectangle(frame, tbox.box, CC_HUMAN_YELLOW, thick, lineType);
        } else if (tbox.class_idx == 1) {
            putText(frame, ss.str(), pt_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, CC_FACE_GREEN, 1, 8);
            cv::rectangle(frame, tbox.box, CC_FACE_GREEN, thick, lineType);
        } else {
            putText(frame, ss.str(), pt_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, CC_HAND_BLUE, 1, 8);
            cv::rectangle(frame, tbox.box, CC_HAND_BLUE, thick, lineType);
        }
    }
}

void run_and_draw_with_sort_tracker(SimilarTracker &trker, CheapSort &sort, cv::Mat &frame, int seq, std::vector<YoloBox> &yolo_boxs)
{
    std::vector<TrackingBox> track_boxes;
    TrackingBox t_box;
    cv::Mat frame_bk = frame.clone();

    for (auto ybox : yolo_boxs) {
        t_box.frame = seq;
        t_box.tracking = 0;
        t_box.id = -1;
        t_box.box.x = ybox.left;
        t_box.box.y = ybox.top;
        t_box.box.width = ybox.right - ybox.left;
        t_box.box.height = ybox.bottom - ybox.top;

        t_box.class_idx = ybox.class_idx;
        t_box.confidence = ybox.confidence;
        track_boxes.push_back(t_box);
    }
    std::vector<TrackingBox> track_result = sort.Run(track_boxes);
    draw_tracking_boxes_with_style(frame, track_result, 1, 8);

    //SimilarTracker
    std::vector<TrackingBox> trk_boxes = trker.Run(frame_bk, track_result);
    std::vector<TrackingBox> trk_hidden_boxes = trker.getHiddenBoxes();

    draw_tracking_boxes_with_style(frame_bk, trk_boxes, 1, 8);
    draw_tracking_boxes_with_style(frame_bk, trk_hidden_boxes, cv::FILLED, 8);

    cv::imshow("SimTracker", frame_bk);
}

void drawCvTrackingResult(bool ok, cv::Mat frame, cv::Rect2d cv_box)
{
    // draw cv tracking
    if (ok) {
        cv::rectangle(frame, cv_box, cv::Scalar(255, 0, 0), 3, 1);
    } else {
        std::cout << "cv ng\n";
    }
}

static int s_tracking_class = SORT_CLS_HUMAN;

std::vector<YoloBox> selected_person_face_hand_class(std::vector<YoloBox> &yolo_boxs)
{
    std::vector<YoloBox> selected;
    for (int i = 0; i < yolo_boxs.size(); i++) {
        //SORT_CLS_HUMAN
        //SORT_CLS_FACE
        //SORT_CLS_HAND
        //thresh
        if (yolo_boxs[i].class_idx == s_tracking_class && yolo_boxs[i].confidence >= 0.01f) {
            selected.push_back(yolo_boxs[i]);
        }
    }
    return selected;
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

    if (argc < 4) {
        std::cerr << "Usage: "
                  << argv[0] << " deploy.prototxt network.caffemodel xxx.mov" << std::endl
                  << argv[0] << " deploy.prototxt network.caffemodel /dev/video0" << std::endl;
        return 1;
    }
    if (argc == 5) {
        s_tracking_class = atoi(argv[4]);
        std::cerr << "Track Class: " << s_tracking_class << std::endl;
    }

    FooTracker fooTracker;

    ::google::InitGoogleLogging(argv[0]);

    string model_file = argv[1];
    string trained_file = argv[2];
    string mov_file = argv[3];
    Yolo2 yolo(model_file, trained_file, model_file, model_file);

    CheapSort cheap_sort;
    SimilarTracker sim_tracker;
    imvt_cv_tracking_init();

    int input_tracking_num = -1;

    ProxyReader reader;

    if (!reader.open(mov_file)) {
        return 0;
    }
    std::vector<YoloBox> yolo_boxs;

    while (1) {
        cv::Mat frame;
        cv::Rect2d cv_box;

        if (pause && pause_frame_to_process <= 0) {
            //do nothing
        } else {
            reader.read(frame);

            // If the frame is empty, break immediately
            if (frame.empty()) {
                break;
            }

            cv::resize(frame, frame, cv::Size(416, 416));

            if (show_once) {
                show_once = 0;
                cv::imshow("YoloTracker", frame);
                cv::imshow("SimTracker",  frame);
            }

            if (!reader.isStream() || detect) {
                // Start timer
                double timer = (double)cv::getTickCount();
                std::vector<YoloBox> yolo_boxs = yolo.Run(frame);
                std::vector<YoloBox> selected = selected_person_face_hand_class(yolo_boxs);
#if 0
                runFooTracker(fooTracker, frame, selected);
                drawYoloResult(frame, frame_seq, selected);
                drawFooTracker(fooTracker, frame);
#elif 1
                run_and_draw_with_sort_tracker(sim_tracker, cheap_sort, frame, frame_seq, selected);
#elif 0
                int ok = imvt_cv_tracking_detect(frame, cv_box);
                drawCvTrackingResult(ok, frame, cv_box);
#else
                simple_draw_yolo_result(frame, frame_seq, selected);
#endif

                // Calculate Frames per second (FPS)
                float fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);
                stringstream ss;
                ss.setf(std::ios::fixed);
                ss.precision(2);
                ss << "FPS:" << fps << " SEQ:" << frame_seq;
                putText(frame, ss.str(), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, 8);
            }

            cv::imshow("YoloTracker",  frame);

            if (pause_frame_to_process > 0) { pause_frame_to_process--;}
            frame_seq++;
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
        } else if (c == 13 /* Enter */) {
            if (input_tracking_num >= 0) {
                sim_tracker.trackById(input_tracking_num);  //select tracker object
                input_tracking_num = -1;
            }
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
        } else if (c == 115 /* 's' */) {
            cv::Rect2d bbox = selectROI("YoloTracker", frame, false);
            imvt_cv_tracking_enable(1, bbox.x, bbox.y, bbox.width, bbox.height);
            detect = 1; //start tracking
        }

    }

    reader.close();

    // Closes all the frames
    cv::destroyAllWindows();
    imvt_cv_tracking_deinit();

    return 0;
}
