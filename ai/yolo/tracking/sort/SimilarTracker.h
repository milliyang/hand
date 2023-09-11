#pragma once

#include "sort_def.h"
#include "SimilarObj.h"

#define SIM_TRK_ID_RESET_NUM            (30*30) //abount 30s

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
    void find_safe_boxes(std::vector<int> &safe, std::vector<TrackingBox> &tboxes);
    void insert_new_object(cv::Mat &frame, TrackingBox &tbox);
    void check_and_remove_object(void);
    void remove_sort_id(int sort_id);

    void debug_image_similar(SimilarObj &obj, cv::Mat &frame, TrackingBox &tbox);
    void debug_check_box_valid(std::vector<TrackingBox> &tboxes);

private:
    int         no_obj_no_box_counter_;
    int         main_trk_id_;
    uint32_t    uuid_;
    int         cur_frame_seq_;
    IntMap      obj_id_map_;       //sort_id -> obj_id_

    std::vector<TrackingBox>    cur_tboxes_;
    std::vector<SimilarObj>     objects_;

    uint8_t     last_obj_cls_counter_[SORT_YOLO_CLASS_NUM];
    uint8_t     curr_obj_cls_counter_[SORT_YOLO_CLASS_NUM];
};
