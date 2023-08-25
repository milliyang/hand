#include "CheapSort.h"

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

void CheapSort::KalmanGlobalReset(void)
{
    KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.
}

CheapSort::CheapSort()
{
    // 3. update across frames
    frame_count = 0;
    max_age = 1;
    min_hits = 3;
    iouThreshold = 0.3;
}

CheapSort::~CheapSort()
{
}

void CheapSort::Clear(void)
{
    trackers.clear();
}

void CheapSort::Init(vector<TrackingBox> t_boxes)
{
    Clear();
    total_frames = 0;
    total_time = 0.0;

    // initialize kalman trackers using first detections.
    for (unsigned int i = 0; i < t_boxes.size(); i++) {
        KalmanTracker trk = KalmanTracker(t_boxes[i].box, t_boxes[i].class_idx, t_boxes[i].class_name, t_boxes[i].confidence);
        trackers.push_back(trk);
    }

    // get trackers' output
    tracking_result.clear();
    for (auto it = trackers.begin(); it != trackers.end();) {
        TrackingBox res;
        res.box = (*it).get_state();
        res.id = (*it).m_id + 1;
        res.frame = frame_count;
        res.class_name = (*it).class_name_;
        res.confidence = (*it).class_confidence_;
        res.class_idx = (*it).class_idx_;
        tracking_result.push_back(res);
        it++;
    }
    // // output the first frame detections
    // for (unsigned int id = 0; id < t_boxes.size(); id++) {
    //     TrackingBox tb = t_boxes[id];
    //     std::cout << tb.frame << "," << id + 1 << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << ",1,-1,-1,-1" << endl;
    // }
}

vector<TrackingBox> CheapSort::Run(vector<TrackingBox> t_boxes)
{
    frame_count++;

    // variables used in the for-loop
    vector<cv::Rect_<float>> predictedBoxes;
    vector<vector<float>> iouMatrix;
    vector<int> assignment;
    set<int> unmatchedDetections;
    set<int> unmatchedTrajectories;
    set<int> allItems;
    set<int> matchedItems;
    vector<cv::Point> matchedPairs;
    vector<TrackingBox> frameTrackingResult;
    unsigned int trkNum = 0;
    unsigned int detNum = 0;

    float cycle_time = 0.0;
    int64 start_time = 0;

    if (trackers.size() == 0) {
        Init(t_boxes);
        return tracking_result;
    }

    // 3.1. get predicted locations from existing trackers.
    predictedBoxes.clear();

    for (auto it = trackers.begin(); it != trackers.end();) {
        cv::Rect_<float> pBox = (*it).predict();
        if (pBox.x >= 0 && pBox.y >= 0) {
            predictedBoxes.push_back(pBox);
            it++;
        } else {
            it = trackers.erase(it);
            //cerr << "Box invalid at frame: " << frame_count << endl;
        }
    }

    ///////////////////////////////////////
    // 3.2. associate detections to tracked object (both represented as bounding boxes)
    // dets : t_boxes
    trkNum = predictedBoxes.size();
    detNum = t_boxes.size();

    iouMatrix.clear();
    iouMatrix.resize(trkNum, vector<float>(detNum, 0));

    // compute iou matrix as a distance matrix
    for (unsigned int i = 0; i < trkNum; i++)  {
        for (unsigned int j = 0; j < detNum; j++) {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], t_boxes[j].box);
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
        for (unsigned int n = 0; n < detNum; n++) {
            allItems.insert(n);
        }

        for (unsigned int i = 0; i < trkNum; ++i) {
            matchedItems.insert(assignment[i]);
        }

        set_difference(allItems.begin(), allItems.end(),
                       matchedItems.begin(), matchedItems.end(),
                       insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
    } else if (detNum < trkNum) {
        // there are unmatched trajectory/predictions

        for (unsigned int i = 0; i < trkNum; ++i) {
            // unassigned label will be set as -1 in the assignment algorithm
            if (assignment[i] == -1) {
                unmatchedTrajectories.insert(i);
            }
        }
    } else {
        //;
    }

    //printf("filter out matched with low IOU\n");

    // filter out matched with low IOU
    matchedPairs.clear();
    for (unsigned int i = 0; i < trkNum; ++i) {
        // pass over invalid values
        if (assignment[i] == -1) {
            continue;
        }
        if (1 - iouMatrix[i][assignment[i]] < iouThreshold) {
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        } else {
            matchedPairs.push_back(cv::Point(i, assignment[i]));
        }
    }

    ///////////////////////////////////////
    // 3.3. updating trackers

    // update matched trackers with assigned detections.
    // each prediction is corresponding to a tracker
    int detIdx, trkIdx;
    for (unsigned int i = 0; i < matchedPairs.size(); i++) {
        trkIdx = matchedPairs[i].x;
        detIdx = matchedPairs[i].y;
        trackers[trkIdx].update(t_boxes[detIdx].box);
    }

    // create and initialise new trackers for unmatched detections
    for (auto umd : unmatchedDetections) {
        KalmanTracker tracker = KalmanTracker(t_boxes[umd].box, t_boxes[umd].class_idx, t_boxes[umd].class_name, t_boxes[umd].confidence);
        trackers.push_back(tracker);
    }

    // get trackers' output
    tracking_result.clear();
    for (auto it = trackers.begin(); it != trackers.end();) {
        if (((*it).m_time_since_update < 1) &&
            ((*it).m_hit_streak >= min_hits || frame_count <= min_hits)) {
            TrackingBox res;
            res.box = (*it).get_state();
            res.id = (*it).m_id + 1;
            res.frame = frame_count;
            res.class_name = (*it).class_name_;
            res.confidence = (*it).class_confidence_;
            res.class_idx = (*it).class_idx_;
            tracking_result.push_back(res);
            ++it;
        } else {
            ++it;
        }

        //bugfix
        //remove dead tracklet
        // if (it != trackers.end() && (*it).m_time_since_update > max_age) {
        //     it = trackers.erase(it);
        // }
    }
#if 1
    for (auto it = trackers.begin(); it != trackers.end();) {
        // remove dead tracklet
        if ((*it).m_time_since_update > max_age) {
            it = trackers.erase(it);
        } else {
            ++it;
        }
    }
#endif
    cycle_time = (float)(cv::getTickCount() - start_time);
    total_time += cycle_time / cv::getTickFrequency();

    // for (auto tb : tracking_result) {
    //     std::cout << "id:" << tb.id << "," << tb.class_name
    //         << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height << endl;
    // }
    return tracking_result;
}
