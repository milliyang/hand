#include<vector>

#include "KalmanTracker.h"
#include "moving_object.h"

#include "opencv2/opencv.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/core/ocl.hpp"

#define LOG_TAG "object"
#include "log.h"

#define NEWLINE    ""
#define DEBUG_LOG

#define ENABLE_PADDING
#define YOLO_PADDING            (0.1f)
#define YOLO_THREHHOLD          (0.1f)
#define YOLO_VALID_CNT          (3)

#define FRAME_FPS               (10)
#define IMAGE_WIDTH             (416)

#define KCF_MAX_OBJECT_SIZE                 (200*200)
#define KCF_UPDATE_INTERVAL                 (5)
#define KCF_THE_SAME_IOU                    (0.9f)

#define LIFE_TIME_NO_MATCH_KEY              (2 * FRAME_FPS) //key tracking object
#define LIFE_TIME_NO_MATCH_KEY_BORDER       (1 * FRAME_FPS) //key tracking object
#define LIFE_TIME_NO_MATCH_KEY_HANDHELD     (10)            //key tracking object
#define LIFE_TIME_NO_MATCH_KEY_SHAKING      (5)             //key tracking object

#define LIFE_TIME_NO_MATCH_NON_KEY          (1 * FRAME_FPS) //not tracking object
#define LIFE_TIME_ONLY_KCF_DETECTION        (20)
#define LIFE_TIME_YOLO_NO_FOUND             (7)

#define STABLE_CNT_FOR_KCF                  (4)
#define STABLE_CNT_FOR_AF_TRACKER           (5 * FRAME_FPS)

#define OBJECT_DISTANCE_CLOSE               (100)
#define OBJECT_DISTANCE_FALL_APART          (10)

#define IOU_AF_TRACKING_SMOOTH                (0.85f)
#define IOU_AF_TRACKING_SMOOTH_HANDHELD       (0.90f)
#define IOU_AF_TRACKING_SMOOTH_SHAKING        (0.95f)

static bool is_valid_double(double x)
{
    return x*0.0==0.0;
}

// Computes IOU between two bounding boxes  (IOU == area of Overlap / area of Union)
static double GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area(); // & means AND, not the overlap area.
    float un = bb_test.area() + bb_gt.area() - in;
    if (un < DBL_EPSILON) {
        return 0;
    }
    return (double)(in / un);
}

MovingObject::MovingObject(Info info, int frameSeq)
{
    //status
    info_ = info;
    lifeCnt_ = 0;
    frameSeq_ = frameSeq;
    frameSeqSys_ = frameSeq;
    kalmanPredictCnt_ = 0;
    onlyKcfMatchCnt_ = 0;
    objectNotMovingCnt_ = 0;

    multiple_ = MULTI_NONE;
    multipleUseKcf_ = false;
    match_ = MATCH_NONE;
    multipleOverlayId_ = -1;
    multipleApartCnt_ = 0;

    kalman_ = KalmanTracker(info.rect, 0, 0);

    kcfInitFrameSeq_ = 0;

    //setup KCF tracking param
    kcfParam_.detect_thresh = 0.5f;
    kcfParam_.sigma = 0.2f;
    kcfParam_.lambda = 0.0001f;
    //kcfParam_.interp_factor = 0.075f;
    kcfParam_.interp_factor = 0.0f; //do not learn for train.

    kcfParam_.output_sigma_factor = 1.0f / 16.0f;
    kcfParam_.resize = true;
    kcfParam_.max_patch_size = 80 * 80;
    kcfParam_.split_coeff = true;
    kcfParam_.wrap_kernel = false;
#if 1
    kcfParam_.desc_npca = cv::TrackerKCF::GRAY;
    kcfParam_.desc_pca = cv::TrackerKCF::CN;
#else
    kcfParam_.desc_npca = 0;
    kcfParam_.desc_pca = cv::TrackerKCF::GRAY | cv::TrackerKCF::CN;
#endif
    //feature compression
    kcfParam_.compress_feature = true;
    kcfParam_.compressed_size = 2;
    kcfParam_.pca_learning_rate = 0.15f;

    imuStatus_ = IMU_STILL;
    smoothIouThresh_ = IOU_AF_TRACKING_SMOOTH;
}

void MovingObject::updateImuStatus(int st)
{
    imuStatus_ = st;
    //update IOU threshold

    if (imuStatus_ == IMU_STILL) {
        smoothIouThresh_ = IOU_AF_TRACKING_SMOOTH;
    } else if (imuStatus_ == IMU_HANDHELD) {
        smoothIouThresh_ = IOU_AF_TRACKING_SMOOTH_HANDHELD;
    } else {
        smoothIouThresh_ = IOU_AF_TRACKING_SMOOTH_SHAKING;
    }
}

void MovingObject::update(int frameSeq, cv::Mat &mat, std::vector<Info> &infos)
{
#ifdef DEBUG_LOG
    LOGI("%s id:%d " NEWLINE, __FUNCTION__, info_.id);
#endif
    Info info;
    match_ = MATCH_NONE;

    cv::Rect_<float> pos = kalmanPredict();
    std::vector<Info>::iterator itInfo;
    frameSeqSys_ = frameSeq;

    /* TODO: kalman bug? sometime nan value return */
    //assert(is_valid_double(pos.x));
    if (!is_valid_double(pos.x)) {
        pos = info_.rect;
    }
#if 0
    if (multiple_ & MULTI_OVERLAP) {
        //when two object overlay, objects should use kalman to predict next position
    }
#endif
    if (multipleUseKcf_) {
        itInfo = findMaxIOU(info_.rect, infos);
        multipleUseKcf_ = false;
    } else if (kalmanPredictCnt_ == 0) {
        itInfo = findMaxIOU(pos, infos);
    } else {
        /* use kalman x coordinate to predict */
        float center = pos.x + pos.width/2.0f;
        pos.width = info_.rect.width;
        pos.height = info_.rect.height;
        pos.y = info_.rect.y;
        pos.x = center - info_.rect.width/2.0f;
        if (pos.x < 0) {
            pos.x = 0;
        }
        itInfo = findMaxIOU(pos, infos);
    }

    if (multiple_ >= MULTI_FALL_APART) {
        multipleApartCnt_++;
        /* only use KCF tracking continually for max 3 frame*/
        if (multipleApartCnt_ >= 3) {
#ifdef DEBUG_LOG
            LOGW("multiple fall apart done\n");
#endif
            multiple_ = MULTI_NONE;
            multipleApartCnt_ = 0;
        }
    }

    if (itInfo != infos.end()) {
        info = *itInfo;
        infos.erase(itInfo);
#ifdef DEBUG_LOG
        LOGI("%s id:%d match erase" NEWLINE, __FUNCTION__, info_.id);
#endif
        match_ |= MATCH_IOU;

        bool stable = smoothYoloRect(info);
        if (stable) {
            objectNotMovingCnt_++;
        } else {
            objectNotMovingCnt_ = 0;
        }

        updateKalmanUsingTheSmoothYoloRect(info);

        if (objectNotMovingCnt_ > STABLE_CNT_FOR_KCF) {
            if (multiple_ < MULTI_OVERLAP && multiple_ >= MULTI_CLOSE) {
                // MULTI_CLOSE or MULTI_NONE
                checkAndUpdateKcfTemplate(mat);
            }
        }

        // save and fast return
        frameSeq_ = frameSeq;
        lifeCnt_++;
        info_.rect = info.rect;
        kalmanPredictCnt_ = 0;

        assert(is_valid_double(info.rect.x));
#ifdef DEBUG_LOG
        LOGI("%s id:%d multiple_ %d" NEWLINE, __FUNCTION__, info_.id, multiple_);
#endif
        if (multiple_ == MULTI_NONE) {
            return;
        }
    }

    if (!isYoloDetected()) {
        return;
    }

#if 0
    // TEST KCF SPEED
    // when disable auto resize, may drop to 3 fps   (60*200)
    // when enable auto resize,  average to  6 fps   (60*200)
    int xmatch;
    if (isKey() && isKcfEnable()) {
        xmatch = runKcfTracking(mat, info.rect);
        LOGI("TEST KCF, match:%d" NEWLINE, xmatch);
    }
#endif

    // No IOU MATCH
    if (match_ == MATCH_NONE) {
#ifdef DEBUG_LOG
        LOGI("%s id:%d No IOU MATCH" NEWLINE, __FUNCTION__, info_.id);
#endif
        //use kalman predict location
        info_.rect = pos;
        kalmanPredictCnt_++;
    }
}

bool MovingObject::shouldBeRemoved(void)
{
    bool remove = false;
    int loss_thresh;
    if (!isYoloDetected() && (frameSeqSys_ - frameSeq_) > LIFE_TIME_YOLO_NO_FOUND) {
#ifdef DEBUG_LOG
        if (info_.id == 0) {
            LOGW("%s: line:%d remove key object" NEWLINE, __FUNCTION__, __LINE__);
        }
#endif
        remove = true;
    }

    if (isKey()) {
        // a. normal
        loss_thresh = LIFE_TIME_NO_MATCH_KEY;

        // b. remove faster when close to border
        if (info_.rect.area() < (30 * 50)) {
            loss_thresh = LIFE_TIME_NO_MATCH_KEY_BORDER; //too small
        } else if (info_.rect.area() < (120*200))  {
            if (info_.rect.x + info_.rect.width >= IMAGE_WIDTH - 30) {
                loss_thresh = LIFE_TIME_NO_MATCH_KEY_BORDER; //right border
            } else if (info_.rect.x < 30) {
                loss_thresh = LIFE_TIME_NO_MATCH_KEY_BORDER; //left border
            }
        }
        // c. remove faster when camera shaking
        if (imuStatus_ == IMU_HANDHELD) {
            loss_thresh = LIFE_TIME_NO_MATCH_KEY_HANDHELD;
        } else if (imuStatus_ == IMU_SHAKING) {
            loss_thresh = LIFE_TIME_NO_MATCH_KEY_SHAKING;
        }
        // use smallest loss thresh
        if (kalmanPredictCnt_ > loss_thresh) {
            remove = true;
        }

        if (onlyKcfMatchCnt_ > LIFE_TIME_ONLY_KCF_DETECTION) {
            remove = true;
        }

        if (multiple_ & MULTI_FALL_APART) {
            if (onlyKcfMatchCnt_ > LIFE_TIME_YOLO_NO_FOUND) {
#ifdef DEBUG_LOG
                LOGW("%s remove tracking object. multiple:%d onlyKcfCnt:%d" NEWLINE, __FUNCTION__, multiple_, onlyKcfMatchCnt_);
#endif
                remove = true;
            }
        }
    } else {
        loss_thresh = LIFE_TIME_NO_MATCH_NON_KEY;

        if (multiple_ & MULTI_FALL_APART) {
            loss_thresh = LIFE_TIME_NO_MATCH_NON_KEY;
        } else if (multiple_ & MULTI_OVERLAP) {
            // Note: should not be too short, because overlaping maybe sometimes >= 30 frame
            // if the tracking object is removed, multiple_ maybe incorrect.
            // test it with the case with longest overlap !!!
            loss_thresh *= 3;
        }

        if (kalmanPredictCnt_ > loss_thresh) {
            remove = true;
        }
    }
    //debug
    if (remove) {
        remove = true;
    }
    return remove;
}

void MovingObject::updateRemoveIds(std::vector<int> &removeIds)
{
    if (!isKey()) {
        return;
    }
    Relation relation;
    for (auto id : removeIds) {
        auto search = multipleRelation_.find(id);
        if (search != multipleRelation_.end()) {
            if (id == multipleOverlayId_) {
                relation = search->second;
                if (relation.status >= MULTI_OVERLAP) {
                    multiple_ |= MULTI_FALL_APART;
                }
            }
            multipleRelation_.erase(search);
        }
    }
}

bool MovingObject::isObjectNotMoving(void)
{
    if (objectNotMovingCnt_ > STABLE_CNT_FOR_AF_TRACKER) {
        return true;
    } else {
        return false;
    }
}

/* this API only for none tracking object*/
void MovingObject::updateMultipleSt(int st)
{
    multiple_ = st;
}

int MovingObject::updateObjectRelation(Info &info)
{
    Relation relation = {0, 0, 0, 0};
    int distance = 0;
    int overlay  = 0;
    float areaRatio;

    auto search = multipleRelation_.find(info.id);
    if (search != multipleRelation_.end()) {
        relation = search->second;
    } else {
#if 1
        areaRatio = info.rect.area() / info_.rect.area();
        if (areaRatio > 1) {
            areaRatio = 1 / areaRatio;
        }
        // ignore object with different size scale. on the first detection
        if (areaRatio < 0.2) {
            return MULTI_NONE;
        }
#endif
    }

    cv::Rect2f in = info_.rect & info.rect;
    if (in.area() > 0) {
        overlay = (int)in.width;
    } else {
        overlay = 0;
        float centerA = info_.rect.x + info_.rect.width / 2;
        float centerB = info.rect.x + info.rect.width / 2;
        distance = (int) std::abs(centerA - centerB);
    }

    if (relation.status & MULTI_FALL_APART) {
        relation.cnt++;
        if (relation.cnt > 10) {
            relation.status = MULTI_NONE; //clear
            relation.cnt = 0;
        }
    } else if (relation.status & MULTI_OVERLAP) {
        // check fall apart
        if (overlay == 0 && distance >= OBJECT_DISTANCE_FALL_APART) {
            relation.status |= MULTI_FALL_APART;
        }
    } else if (relation.status & MULTI_CLOSE) {
        // check overlap
        if (overlay > 0) {
            relation.status |= MULTI_OVERLAP;
        } else if (distance > OBJECT_DISTANCE_CLOSE) {
            //return to none status
            relation.status = MULTI_NONE;
        }
    } else { /*(relation.status == MULTI_NONE)*/
        if (distance > 0 && distance < OBJECT_DISTANCE_CLOSE) {
            relation.status |= MULTI_CLOSE;
        }
    }

    relation.distance = distance;
    relation.overlay = overlay;
    multipleRelation_[info.id] = relation;
    return relation.status;
}

cv::Rect MovingObject::getPosition(void)
{
    return info_.rect;
}

void MovingObject::setKeyId(void)
{
    info_.id = 0;
}

void MovingObject::resetId(int id)
{
    info_.id = id;
}

Info MovingObject::getInfo(void)
{
    return info_;
}

int MovingObject::getId(void)
{
    return info_.id;
}

void MovingObject::checkObjectDistance(cv::Mat &mat, std::vector<MovingObject> &objects)
{
    Info info;
    int maxId = -1;
    int maxStatus = 0;
    for (auto it = objects.begin(); it != objects.end(); it++) {
        // skip id == 0
        if (it == objects.begin()) {
            continue;
        }
        // skip invalid object
        if (!(*it).isYoloDetected()) {
            continue;
        }
        info = (*it).getInfo();
        int st = updateObjectRelation(info);
        (*it).updateMultipleSt(st);

        if (st > maxStatus) {
            maxId = info.id;
            maxStatus = st;
        }
    }

    /*
     * if one object is falling apart:
     *  1. do not update multiple_
     *  2. uss post KCF tracking
     */
    if (multiple_ >= MULTI_FALL_APART || maxStatus >= MULTI_FALL_APART) {
        if (maxStatus >= MULTI_FALL_APART) {
            multiple_ = maxStatus;
            multipleOverlayId_ = maxId;
        }
        postKcfTracking(mat);
    } else {
        // use max status
        multiple_ = maxStatus;
        multipleOverlayId_ = maxId;
    }
}

void MovingObject::postKcfTracking(cv::Mat &mat)
{
    cv::Rect2f rect;

    // no need do twice
    if (match_ & MATCH_KCF) {
        return;
    }

    if (isKcfEnable()) {
        bool kcfMatch = runKcfTracking(mat, rect);
        if (kcfMatch) {
            match_ |= MATCH_KCF;
#ifdef DEBUG_LOG
            LOGI("%s id:%d POST KCF tracking match" NEWLINE, __FUNCTION__, info_.id);
#endif
            if (match_ & MATCH_IOU) {
                cv::Rect2f in = rect & info_.rect;
                if (in.area() > 0) {
                    //use IOU
#ifdef DEBUG_LOG
                    LOGI("%s id:%d both IOU and KCF math, overlap. use IOU" NEWLINE, __FUNCTION__, info_.id);
#endif
                    return;
                }
            }

            //save
            lifeCnt_++;
            frameSeq_ = frameSeqSys_;
            info_.rect = rect;
            kalmanPredictCnt_ = 0;
            multipleUseKcf_ = true;
#ifdef DEBUG_LOG
            LOGI("%s id:%d both IOU and KCF match, not overlap. use KCF" NEWLINE, __FUNCTION__, info_.id);
#endif
        }
    }
}

bool MovingObject::isKey()
{
    return info_.id == 0;
}

bool MovingObject::isYoloDetected()
{
    return lifeCnt_ >= YOLO_VALID_CNT;
}

bool MovingObject::isKcfEnable()
{
    return kcf_ != nullptr;
}

void MovingObject::checkAndUpdateKcfTemplate(cv::Mat &frame)
{
    /* only the key tracking object use KCF*/
    if (!isKey()) {
        return;
    }
    /* yolo stable (detect more than 5 times) */
    if (!isYoloDetected()) {
        return;
    }
    /* already init thie round*/
    if (kcfInitFrameSeq_ == frameSeq_) {
        return;
    }
    if (info_.rect.area() > KCF_MAX_OBJECT_SIZE) {
#ifdef DEBUG_LOG
        LOGI("%s TOO LARGE, SKIP KCF" NEWLINE, __FUNCTION__);
#endif
        return;
    }
    /* only update KCF every n frame */
    if (kcf_ != nullptr) {
        if ((frameSeqSys_ - kcfInitFrameSeq_) < KCF_UPDATE_INTERVAL) {
            return;
        }
        //if roi every close. skip
        double iou = GetIOU(kcfInitRect_, info_.rect);
        if (iou > KCF_THE_SAME_IOU) {
#ifdef DEBUG_LOG
            LOGW("%s SKIP KCF TEMPLATE, the same" NEWLINE, __FUNCTION__);
#endif
            return;
        }
    }
#ifdef DEBUG_LOG
    LOGW("%s INIT KCF TEMPLATE" NEWLINE, __FUNCTION__);
#endif
    kcf_ = cv::TrackerKCF::create(kcfParam_);
    kcf_->init(frame, info_.rect);
    kcfInitFrameSeq_ = frameSeqSys_;
    kcfInitRect_ = info_.rect;
    runKcfTracking(frame, kcfInitRect_);
}

bool MovingObject::runKcfTracking(cv::Mat &frame, cv::Rect2f &rect)
{
    if (kcf_ == nullptr) {
        return false;
    }

    cv::Rect2i box; //fix for #4.5.5
    bool ok = kcf_->update(frame, box);
    if (ok) {
        rect = box;
    }
    return ok;
}

std::vector<Info>::iterator MovingObject::findMaxIOU(cv::Rect_<float> rect, std::vector<Info> &infos)
{
    double max = 0;
    double iou;
    std::vector<Info>::iterator cur = infos.end();
    for (auto it = infos.begin(); it != infos.end(); it++) {
        iou = GetIOU(rect, (*it).rect);
        if (iou > max) {
            cur = it;
            max = iou;
        }
    }
#ifdef DEBUG_LOG
    LOGI("%s id:%d final pos:%d\n" NEWLINE, __FUNCTION__, info_.id, (int)(cur - infos.begin()));
#endif
    return cur;
}

cv::Rect_<float> MovingObject::kalmanPredict(void)
{
    return kalman_.predict();
}

bool MovingObject::smoothYoloRect(Info &info)
{
    double iou = GetIOU(info_.rect, info.rect);
#if 0
    double areaRatio = 1.0 - info_.rect.area() / info.rect.area();
    double ratio = 1.0 - (info_.rect.width / info_.rect.height)  / (info.rect.width / info.rect.height);
    if (areaRatio < 0) {
        areaRatio *= -1;
    }
    if (ratio < 0) {
        ratio *= -1;
    }
    LOGI("%s id:%d iou:%f areaRatio:%f ratio:%f" NEWLINE, __FUNCTION__, info_.id, iou, areaRatio, ratio);
    // when the size are the same, but the rect is moving. use smaller iou
    if (ratio < 0.1f && areaRatio < 0.1f) {
        if (iou < 0.9f) {
            return false;
        }
    }
#endif
    if (iou > smoothIouThresh_) {
        info = info_;
        return true;
    }
    return false;
}

void MovingObject::updateKalmanUsingTheSmoothYoloRect(Info &info)
{
    kalman_.update(info.rect);
}
