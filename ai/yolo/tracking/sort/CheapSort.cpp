#include "CheapSort.h"

#define LOG_TAG "sort"
#ifdef CONFIG_SPDLOG
#include "log.h"
#else
#include "sys/log/imvt_log.h"
#endif

bool check_box_valid(const cv::Rect_<float> &bb)
{
    if (bb.x >= 0 && bb.y >= 0) {
        return true;
    } else {
        //bb.x == nan
        return false;
    }
}

// Computes IOU between two bounding boxes
float GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;
    if (un < DBL_EPSILON) {
        return 0;
    }
    return (float)(in / un);
}

void CheapSort::KalmanGlobalResetId(void)
{
    KalmanTracker::kf_count = 1; // tracking id relies on this, so we have to reset it in each seq.
}

CheapSort::CheapSort()
{
    frame_count_    = 0;
    max_age_        = SORT_EXPIRE_MAX_AGE;
    min_hits_       = 2;
    iou_threshold_  = SORT_IOU_THRESHOLD;
}

CheapSort::~CheapSort()
{
}

void CheapSort::Clear(void)
{
    trackers_.clear();
}

void CheapSort::Init(vector<TrackingBox> t_boxes)
{
    Clear();
    total_time_ = 0.0;

    // initialize kalman trackers using first detections.
    for (uint32_t i = 0; i < t_boxes.size(); i++) {
        KalmanTracker trk = KalmanTracker(t_boxes[i].box, t_boxes[i].class_idx, t_boxes[i].confidence);
        trackers_.push_back(trk);
    }

    // get trackers' output
    tracking_result_.clear();
    for (auto it = trackers_.begin(); it != trackers_.end();) {
        TrackingBox res;
        res.box = (*it).get_state();
        res.tracking = 0;
        res.id = (*it).m_id;
        res.frame = frame_count_;
        res.confidence = (*it).confidence_;
        res.class_idx = (*it).class_idx_;
        tracking_result_.push_back(res);
        it++;
    }
}

vector<TrackingBox> CheapSort::Run(vector<TrackingBox> t_boxes)
{
    frame_count_++;

    // variables used in the for-loop
    std::vector<cv::Rect_<float>> predictedBoxes;
    std::vector<uint8_t>          predictedBoxesClass;
    std::vector<vector<float>> iouMatrix;
    std::vector<int> assignment;
    std::set<int> unmatchedDetections;
    std::set<int> unmatchedTrajectories;
    std::set<int> allItems;
    std::set<int> matchedItems;
    std::vector<cv::Point> matchedPairs;
    uint32_t trkNum = 0;
    uint32_t detNum = 0;

    float cycle_time = 0.0;
    int64 start_time = 0;

    if (trackers_.size() == 0) {
        if (t_boxes.size() == 0) {
            no_tracker_count_++;
            if (no_tracker_count_ >= SORT_RESET_ID_FRAME_NUM) {
                KalmanGlobalResetId();
                no_tracker_count_ = 0;
            }
        }
        Init(t_boxes);
        return tracking_result_;
    }
    no_tracker_count_ = 0;

    // 3.1. get predicted locations from existing trackers_.
    predictedBoxes.clear();
    predictedBoxesClass.clear();

    for (auto it = trackers_.begin(); it != trackers_.end();) {
        cv::Rect_<float> pdBox = (*it).predict();
        if (check_box_valid(pdBox)) {
            predictedBoxes.push_back(pdBox);
            predictedBoxesClass.push_back((*it).class_idx_);
            it++;
        } else {
            it = trackers_.erase(it);
        }
    }

    // 3.2. associate detections to tracked object (both represented as bounding boxes)
    trkNum = predictedBoxes.size();
    detNum = t_boxes.size();

    //bugfix: Leo
    if (trkNum <= 0) {
        Init(t_boxes);
        return tracking_result_;
    }

    iouMatrix.clear();
    iouMatrix.resize(trkNum, vector<float>(detNum, 0));

    // compute iou matrix as a distance matrix
    for (uint32_t i = 0; i < trkNum; i++)  {
        for (uint32_t j = 0; j < detNum; j++) {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            if (predictedBoxesClass[i] == t_boxes[j].class_idx) {
                iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], t_boxes[j].box);
            } else {
                iouMatrix[i][j] = 1 - 0; //Leo fix: diff class always 0 iou
            }
        }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    assignment.clear();
    HungAlgo.Solve(iouMatrix, assignment);

    // find matches, unmatched_detections and unmatched_predictions
    unmatchedTrajectories.clear();
    unmatchedDetections.clear();
    allItems.clear();
    matchedItems.clear();

    //	there are unmatched detections
    if (detNum > trkNum)  {
        for (uint32_t n = 0; n < detNum; n++) {
            allItems.insert(n);
        }
        for (uint32_t i = 0; i < trkNum; ++i) {
            matchedItems.insert(assignment[i]);
        }
        set_difference(allItems.begin(), allItems.end(),
                       matchedItems.begin(), matchedItems.end(),
                       insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
    } else if (detNum < trkNum) {
        // there are unmatched trajectory/predictions
        for (uint32_t i = 0; i < trkNum; ++i) {
            // unassigned label will be set as -1 in the assignment algorithm
            if (assignment[i] == -1) {
                unmatchedTrajectories.insert(i);
            }
        }
    }

    // filter out matched with low IOU
    matchedPairs.clear();
    for (uint32_t i = 0; i < trkNum; ++i) {
        // pass over invalid values
        if (assignment[i] == -1) {
            continue;
        }
        if (1 - iouMatrix[i][assignment[i]] < iou_threshold_) {
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        } else {
            matchedPairs.push_back(cv::Point(i, assignment[i]));
        }
    }

    // 3.3. updating trackers
    // update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker
    int detIdx, trkIdx;
    for (uint32_t i = 0; i < matchedPairs.size(); i++) {
        trkIdx = matchedPairs[i].x;
        detIdx = matchedPairs[i].y;
        trackers_[trkIdx].update(t_boxes[detIdx].box);
        trackers_[trkIdx].confidence_ = t_boxes[detIdx].confidence;
    }

    // create and initialise new trackers for unmatched detections
    for (auto umd : unmatchedDetections) {
        KalmanTracker tracker = KalmanTracker(t_boxes[umd].box, t_boxes[umd].class_idx, t_boxes[umd].confidence);
        trackers_.push_back(tracker);
    }

    // get trackers' output
    tracking_result_.clear();
    for (auto & trker : trackers_) {
        if ((trker.m_time_since_update < 1) &&
            (trker.m_hit_streak >= min_hits_ || frame_count_ <= min_hits_)) {
            TrackingBox tbox;
            tbox.box         = trker.get_state();
            tbox.tracking  = 0;
            tbox.id          = trker.m_id;
            tbox.frame       = frame_count_;
            tbox.confidence  = trker.confidence_;
            tbox.class_idx   = trker.class_idx_;
            tracking_result_.push_back(tbox);
        }
    }

#if 1
    for (auto it = trackers_.begin(); it != trackers_.end();) {
        // remove dead tracker
        if ((*it).m_time_since_update > max_age_) {
            //LOGD("expire id:%d cls:%d\n", (*it).m_id, (*it).class_idx_);
            it = trackers_.erase(it);
        } else {
            ++it;
        }
    }
#endif

#if 1
    if (trackers_.size() >= SORT_NUM) {
        LOGW("[warn] trackers:%d\n", (int) trackers_.size());
    }
#endif

    cycle_time = (float)(cv::getTickCount() - start_time);
    total_time_ += cycle_time / cv::getTickFrequency();

    return tracking_result_;
}
