#include<vector>

#include "moving_object.h"
#include "foo_tracker.h"

#define LOG_TAG "fooTracker"
#include "log.h"
#define NEWLINE ""
#define xDEBUG_LOG

#define THRESHOLD_STILL     (50)
#define THRESHOLD_HANDHELD  (250)

FooTracker::FooTracker()
{
    mode_ = AF;
    uniqueId_ = 0;
    frameSeq_ = 0;

    imuAverage10_ = -1;
    imuAverage20_ = -1;
    imuAverage60_ = -1;
}

void FooTracker::updateYoloResult(cv::Mat &mat, std::vector<Info> &yoloInfos)
{
    yoloLast_ = yoloInfos;

    preTracking(mat);
    checkMuipleTracking(mat);
    frameSeq_++;
}

bool FooTracker::getResult(cv::Rect_<float> &result)
{
    if (objects_.empty()) {
        return false;
    } else {
        result = objects_.front().getInfo().rect;
        return true;
    }
}

std::vector<Info> FooTracker::getAllResult(void)
{
    std::vector<Info> results;
    for (auto obj : objects_) {
        Info info = obj.getInfo();
        results.push_back(info);
    }
    return results;
}

void FooTracker::updateImuInfo(int angle_xyz)
{
    int status;
    if (angle_xyz < 0) {
        angle_xyz = 0;
    }
    if (imuAverage10_ < 0) {
        imuAverage10_ = angle_xyz;
        imuAverage20_ = angle_xyz;
        imuAverage60_ = angle_xyz;
    } else {
        imuAverage10_ = (imuAverage10_ * 9 + angle_xyz ) / 10;
        imuAverage20_ = (imuAverage20_ * 19 + angle_xyz ) / 20;
        imuAverage60_ = (imuAverage60_ * 59 + angle_xyz ) / 60;
    }

    if (imuAverage10_ < THRESHOLD_STILL && imuAverage20_ < THRESHOLD_STILL && imuAverage60_ < THRESHOLD_STILL) {
        status = MovingObject::IMU_STILL;
    } else if (imuAverage10_ < THRESHOLD_HANDHELD && imuAverage20_ < THRESHOLD_HANDHELD && imuAverage60_ < THRESHOLD_HANDHELD) {
        status = MovingObject::IMU_HANDHELD;
    } else {
        status = MovingObject::IMU_SHAKING;
    }

    imuStatus_ = status;
#ifdef DEBUG_LOG
    LOGW("angle_xyz:%d avg10:%d avg20:%d avg60:%d\n", angle_xyz, imuAverage10_, imuAverage20_, imuAverage60_);
#endif
}

void FooTracker::preTracking(cv::Mat &mat)
{
    std::vector<Info> infos = yoloLast_;
    std::vector<int> removedIds;
    bool trackingOjbectRemoved = false;

    for (auto it = objects_.begin(); it != objects_.end(); it++) {
        (*it).updateImuStatus(imuStatus_);
        (*it).update(frameSeq_, mat, infos);
    }

    //assign new object to the left result
    for (auto it = infos.begin(); it != infos.end();) {
        (*it).id = uniqueId_;
        MovingObject obj(*it, frameSeq_);
        objects_.push_back(obj);
        it = infos.erase(it);
        uniqueId_++;
    }

    //remove object on condition
    for (auto it = objects_.begin(); it != objects_.end();) {
        if ((*it).shouldBeRemoved()) {
            removedIds.push_back((*it).getId());

            if ((*it).getId() == 0) {
#ifdef DEBUG_LOG
                LOGW("%s remove key obj(%d)" NEWLINE, __FUNCTION__, (*it).getId());
#endif
                trackingOjbectRemoved = true;
            }
            it = objects_.erase(it);
            continue;
        }
        it++;
    }

    // update and reset id
    if (objects_.empty()) {
        uniqueId_ = 0;
    } else {
        if (trackingOjbectRemoved) {
            if (objects_.size() == 1) {
                (*objects_.begin()).setKeyId();
            }

            //TODO:
            // search the closest to be key tracking object.  (x)
            // use center as key tracking object.  (x)
            // use max area
            int maxArea = 0;
            cv::Rect2f rect;

            std::vector<MovingObject>::iterator itMatch = objects_.begin();
            for (auto it = objects_.begin(); it != objects_.end(); it++) {
                rect = (*it).getInfo().rect;
                if (rect.area() > maxArea) {
                    itMatch = it;
                    maxArea = rect.area();
                }
            }
            if (itMatch != objects_.begin()) {
                //do swap
                MovingObject newObj = (*itMatch);
                objects_.erase(itMatch);
                objects_.insert(objects_.begin(), newObj);
            }
            (*objects_.begin()).setKeyId();
        } else {
            if (removedIds.size() > 0) {
                (*objects_.begin()).updateRemoveIds(removedIds);
            }
        }
    }
}

// multi object and overlay happened
void FooTracker::checkMuipleTracking(cv::Mat &mat)
{
    if (objects_.size() <= 1) {
        return;
    }

    auto trackingIt = objects_.begin();
    (*trackingIt).checkObjectDistance(mat, objects_);

#if 1
    //auto switch tracking target
    cv::Rect2f trackingRect = (*trackingIt).getInfo().rect;
    std::vector<MovingObject>::iterator itMatch = objects_.begin();
    for (auto it = objects_.begin(); it != objects_.end(); it++) {
        if (it == objects_.begin()) {
            continue;
        }
        if ((*it).isObjectNotMoving()) {
            cv::Rect2f rect = (*it).getInfo().rect;
            if (rect.area() / trackingRect.area() > 1.2f) {
                //swtich main tracking object
                itMatch = it;
                break;
            }
        }
    }

    if (itMatch != objects_.begin()) {
        LOGI("%s tracking target swtich" NEWLINE, __FUNCTION__);
        //do swap
        objects_.begin()->resetId(uniqueId_++);
        MovingObject newObj = (*itMatch);
        objects_.erase(itMatch);
        newObj.setKeyId();
        objects_.insert(objects_.begin(), newObj);
    }
#endif

}