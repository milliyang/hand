#pragma once
#include<vector>
#include "moving_object.h"

class FooTracker
{

    enum MODE {
        PTZ    = 0, //always tracking the sample object; Human intervention(select)
        AF     = 1, //some times may switch tracking target.
    };

public:
    FooTracker();
    void updateImuInfo(int angle);
    void updateYoloResult(cv::Mat &mat, std::vector<Info> &yoloInfos);
    bool getResult(cv::Rect_<float> &result);
    std::vector<Info> getAllResult(void);

private:
    void preTracking(cv::Mat &mat);
    void checkMuipleTracking(cv::Mat &mat);

    std::vector<Info> yoloLast_;

    int frameSeq_;
    int uniqueId_;                      //key obejct always 0
    std::vector<MovingObject> objects_; //key object always the first
    MODE mode_;

    int imuStatus_;
    int imuAverage10_; //10 frame moving average
    int imuAverage20_; //20 frame moving average
    int imuAverage60_; //60 frame moving average
};