#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include "yolo1.h"
#include "yolo2.h"
#include "yolo2ab.h"
#include "CheapSort.h"
#include "proxy_reader.h"
#include "imvt_cv_tracking.h"

#include "foo_tracker.h"
#include "moving_object.h"

using namespace std;

// ./YoTracker.exe yolo_small_deploy.prototxt yolo_small.caffemodel yolo.mov
// ./YoTracker.exe yolo2_hisi/yolo2.prototxt yolo2_hisi/yolo2.caffemodel yolo.mov
// ./YoTracker.exe yolov2.prototxt yolov2.caffemodel car.mov
// ./YoTracker.exe yolov2.prototxt yolov2.caffemodel person.jpg
// ./YoTracker.exe fake.prototxt fake.caffemodel person.jpg

//  from: 192.168.9.6
// ./YoTracker.exe yolov2.prototxt yolov2.caffemodel tcp://192.168.9.6:5556
// ./YoTracker.exe yolov2.prototxt yolov2.caffemodel tcp://192.168.9.7:5556

//  ./YoTracker.exe yolov3.mini.prototxt yolov3.mini.caffemodel street_cars.jpg

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
        putText(frame, info.str(), cv::Point(5, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2, 8);
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
            putText(frame, ss.str(), pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 250, 0), 2, 8);
        }
    }
}

void drawYoloResult(cv::Mat &frame, int seq, std::vector<YoloBox> &yolo_boxs)
{
    if (yolo_boxs.size() <= 0) {
        std::cout << "not found! frame_seq:" << seq << std::endl;
    } else {
        for (auto ybox : yolo_boxs) {
            std::cout << "Yolo :" << ybox.class_name << " idx:" << ybox.class_idx << " confidence:" << ybox.confidence << " " << ybox.left << " " << ybox.top << " " << ybox.right << " " << ybox.bottom << " frame_seq:" << seq << std::endl;
            //rectangle(Mat& img, Rect rec, const Scalar& color, int thickness=1, int lineType=8, int shift=0 )
            cv::Rect rect(ybox.left, ybox.top, (ybox.right - ybox.left), (ybox.bottom - ybox.top));
            cv::rectangle(frame, rect, cv::Scalar(250, 0, 0));
        }
    }
}

void runAndDrawCheapSortTracking(CheapSort &sort, cv::Mat frame, int seq, std::vector<YoloBox> &yolo_boxs)
{
    // format to CheapSort box
    vector<TrackingBox> track_boxes;
    TrackingBox t_box;
    for (auto ybox : yolo_boxs) {
        t_box.frame = seq;
        t_box.id = -1;
        t_box.box.x = ybox.left;
        t_box.box.y = ybox.top;
        t_box.box.width = ybox.right - ybox.left;
        t_box.box.height = ybox.bottom - ybox.top;

        t_box.class_idx = ybox.class_idx;
        t_box.class_name = ybox.class_name;
        t_box.confidence = ybox.confidence;
        track_boxes.push_back(t_box);
    }
    vector<TrackingBox> track_result = sort.Run(track_boxes);
    cv::Point textOrg;
    string label;

    for (auto result : track_result) {
        std::cout << "Track:" << result.class_name << " idx:" << result.class_idx << " confidence:" << result.confidence << " " << result.box.x << " " << result.box.y << " id:" << result.id << std::endl;
        //rectangle(Mat& img, Rect rec, const Scalar& color, int thickness=1, int lineType=8, int shift=0 )
        //cv::Rect rect(result.xbox);
        cv::rectangle(frame, result.box, cv::Scalar(255, 0, 0));

        // then put the text itself
        textOrg.x = result.box.x;
        textOrg.y = result.box.y > 2 ? result.box.y - 5 : 0;
        stringstream ss;
        ss << "(" << result.id << ") " << result.class_name << " P|" << result.confidence; //result.class_name;
        putText(frame, ss.str(), textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 250, 0), 2, 8);
    }
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

int main(int argc, char **argv)
{
    int ret;
    int detect = 0;
    int source = 0;

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " deploy.prototxt network.caffemodel xxx.mov" << std::endl;
        return 1;
    }

    FooTracker fooTracker;

    ::google::InitGoogleLogging(argv[0]);

    string model_file = argv[1];
    string trained_file = argv[2];
    string mov_file = argv[3];
    Yolo2 yolo(model_file, trained_file, model_file, model_file);

    CheapSort cheap_sort;
    imvt_cv_tracking_init();

    ProxyReader frameReader;

    if (!frameReader.open(mov_file)) {
        return 0;
    }
    std::vector<YoloBox> yolo_boxs;

    int frame_seq = 0;
    while (1) {

        cv::Mat frame;
        cv::Rect2d cv_box;

        frameReader.read(frame);

        // If the frame is empty, break immediately
        if (frame.empty()) {
            break;
        }

#if 0
        if (frame_seq % 1 != 0 || frame_seq < 1) {
            frame_seq++;
            continue;
        }
#endif

        //cv::resize(frame, frame, cv::Size(848, 480));
        cv::resize(frame, frame, cv::Size(416, 416));

        if (!frameReader.isStream() || detect) {
            // Start timer
            double timer = (double)cv::getTickCount();
            //int ok = imvt_cv_tracking_detect(frame, cv_box);
            //std::vector<YoloBox> yolo_boxs;
            yolo_boxs = yolo.Run(frame);
            runFooTracker(fooTracker, frame, yolo_boxs);
            drawYoloResult(frame, frame_seq, yolo_boxs);
            drawFooTracker(fooTracker, frame);
            //runAndDrawCheapSortTracking();
            //drawCvTrackingResult(ok, frame, cv_box);

            // Calculate Frames per second (FPS)
            float fps = cv::getTickFrequency() / ((double)cv::getTickCount() - timer);
            stringstream sfps;
            sfps << "FPS: " << fps;
            putText(frame, sfps.str(), cv::Point(5, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2, 8);
        }

        // Display the resulting frame
        cv::imshow("YoTracker", frame);

        // Press  ESC on keyboard to exit
        char c;
        if (frameReader.isStream()) {
            c = (char)cv::waitKey(25);
        } else {
            c = (char)cv::waitKey(25 * 10000);
        }
        //std::cout << "c:" << (int)c << std::endl;
        if (c == 27) {
            break;
        } else if (c == 116 /* 't' */) {
            detect = !detect;
        } else if (c == 115 /* 's' */) {
            //select roi
            cv::Rect2d bbox;
            bbox = selectROI("YoTracker", frame, false);
            imvt_cv_tracking_enable(1, bbox.x, bbox.y, bbox.width, bbox.height);

            //start tracking
            detect = 1;
        }
        frame_seq++;
    }

    frameReader.close();

    // Closes all the frames
    cv::destroyAllWindows();
    imvt_cv_tracking_deinit();

    return 0;
}