#pragma once

#include "sort_def.h"
#include "SimilarObj.h"

class SimilarTracker
{
public:
    SimilarTracker();
    ~SimilarTracker();

    /*start tracking by ID/Rect */
    void trackById(int id);
    void trackByRect(const RectBox &rect);

    std::vector<TrackingBox> Run(cv::Mat &frame, std::vector<TrackingBox> &tboxes);
    std::vector<TrackingBox> getBoxes(void);
    std::vector<TrackingBox> getHiddenBoxes(void);

private:
    bool check_and_reinit(cv::Mat &frame, std::vector<TrackingBox> &tboxes);
    void find_safe_boxes(std::set<int> &safe, std::vector<TrackingBox> &tboxes);
    void insert_new_object(cv::Mat &frame, TrackingBox &tbox);
    void check_and_remove_object(void);

    void debug_image_ssim(SimilarObj &obj, cv::Mat &frame, TrackingBox &tbox);

private:
    int         main_trk_id_;
    uint32_t    uuid_;
    int         cur_frame_seq_;
    IntMap      obj_id_map_;       //sort_id -> obj_id_

    std::vector<TrackingBox>    cur_tboxes_;
    std::vector<SimilarObj>     objects_;
    uint8_t                     objects_cls_counter_[SORT_YOLO_CLASS_NUM];
};
