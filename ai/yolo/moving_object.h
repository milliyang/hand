#pragma once

#include<vector>
#include<map>

#include "KalmanTracker.h"

#include "opencv2/opencv.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/core/ocl.hpp"

typedef struct info {
    int id;
    cv::Rect_<float> rect;
} Info;

typedef struct relattion {
    int status;   // MULTI_NONE/MULTI_CLOSE/MULTI_OVERLAP/MULTI_FALL_APART
    int overlay;  //
    int distance; //
    int cnt;
} Relation;

class MovingObject
{
    /*
     * the status can only be changed step by step.
     * eg. can't jump the fall apart without overlap status
     * that means: we don't detect unusual situation.
     */
    enum MULTIPLE {
        MULTI_NONE         = (0),
        MULTI_CLOSE        = (1<<0),
        MULTI_OVERLAP      = (1<<1),
        MULTI_FALL_APART   = (1<<2),
    };

    enum MATCH {
        MATCH_NONE = (0),
        MATCH_IOU  = (1<<0),
        MATCH_KCF  = (1<<1),
    };

public:
    enum IMU {
        IMU_STILL         = 0, //(no imu or xyz every small within 1 minute)
        IMU_HANDHELD      = 1, //(xyz small, using gimbal tracking (20second) )
        IMU_SHAKING       = 2, //(xyz large, vibrating (10 second) )
    };

public:
    MovingObject(Info info, int frameSeq);

    void updateImuStatus(int st);
    void update(int frameSeq, cv::Mat &mat, std::vector<Info> &infos);
    void checkObjectDistance(cv::Mat &mat, std::vector<MovingObject> &objects);

    bool shouldBeRemoved(void);
    void updateRemoveIds(std::vector<int> &removeIds);
    bool isObjectNotMoving(void);

    cv::Rect getPosition(void);

    void setKeyId(void);
    void resetId(int id);

    Info getInfo(void);
    int getId(void);

private:
    void postKcfTracking(cv::Mat &mat);
    int  updateObjectRelation(Info &info);
    void updateMultipleSt(int st);
    bool isKey();
    bool isYoloDetected();
    bool isKcfEnable();
    void checkAndUpdateKcfTemplate(cv::Mat &frame);
    bool runKcfTracking(cv::Mat &frame, cv::Rect2f &rect);
    std::vector<Info>::iterator findMaxIOU(cv::Rect_<float> rect, std::vector<Info> &infos);
    cv::Rect_<float> kalmanPredict(void);
    bool smoothYoloRect(Info &info);
    void updateKalmanUsingTheSmoothYoloRect(Info &info);

private:
    int frameSeq_;
    int frameSeqSys_;
    unsigned int lifeCnt_;
    Info info_;
    int match_;
    int multiple_;
    int multipleApartCnt_;
    int multipleOverlayId_;

    bool multipleUseKcf_;
    std::map<int, Relation> multipleRelation_;

    int kalmanPredictCnt_;
    int onlyKcfMatchCnt_;
    int objectNotMovingCnt_;

    cv::Ptr<cv::Tracker> kcf_;
    cv::TrackerKCF::Params kcfParam_;
    int kcfInitFrameSeq_;
    cv::Rect2f kcfInitRect_;
    KalmanTracker kalman_;

    int imuStatus_;
    float smoothIouThresh_;
};