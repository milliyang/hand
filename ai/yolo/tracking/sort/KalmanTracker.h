#pragma once

#ifndef __KalmanTracker_H__
#define __KalmanTracker_H__

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define StateType Rect_<float>

// This class represents the internel state of individual tracked objects observed as bounding box.
class KalmanTracker
{
public:
    KalmanTracker()
    {
        init_kf(StateType());
        m_time_since_update = 0;
        m_hits = 0;
        m_hit_streak = 0;
        m_age = 0;
        m_id = kf_count;
    }
    KalmanTracker(StateType initRect, int class_idx, float confidence)
    {
        init_kf(initRect);
        m_time_since_update = 0;
        m_hits = 0;
        m_hit_streak = 0;
        m_age = 0;
        m_id = kf_count;
        kf_count++;

        class_idx_ = class_idx;
        confidence_ = confidence;
    }

    ~KalmanTracker()
    {
        m_history.clear();
    }

    StateType predict();
    void update(StateType stateMat);

    StateType get_state();
    StateType get_rect_xysr(float cx, float cy, float s, float r);

    static int kf_count;

    int m_time_since_update;
    int m_hits;
    int m_hit_streak;
    int m_age;
    int m_id;

    //leo for yolo
    int          class_idx_;
    float        confidence_;
private:
    void init_kf(StateType stateMat);

    cv::KalmanFilter kf;
    cv::Mat measurement;

    std::vector<StateType> m_history;
};




#endif