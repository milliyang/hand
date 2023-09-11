#include "SimilarTracker.h"

#define LOG_TAG "SimTrker"

#ifdef CONFIG_SPDLOG
#include "log.h"
#else
#include "sys/log/imvt_log.h"
#endif

#define SSIM_FEATURE_SIZE   (8)

// Computes IOU between two bounding boxes
static float GetIOU(const RectBox &bb_test, const RectBox &bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;
    if (un < DBL_EPSILON) {
        return 0;
    }
    return (float)(in / un);
}

static float GetSimpleDist(const RectBox &bb_a, const RectBox &bb_b)
{
    float a_x = bb_a.x + bb_a.width/2.0f;
    float a_y = bb_a.y + bb_a.height/2.0f;
    float b_x = bb_b.x + bb_b.width/2.0f;
    float b_y = bb_b.y + bb_b.height/2.0f;

    a_x -= b_x;
    a_y -= a_y;
    return a_x*a_x + a_y*a_y;
}

SimilarTracker::SimilarTracker(void)
{
    uuid_ = 1;
    main_trk_id_ = -1;
    no_obj_no_box_counter_ = 0;

    memset(last_obj_cls_counter_, 0, sizeof(last_obj_cls_counter_));
}

SimilarTracker::~SimilarTracker(void)
{

}

void SimilarTracker::insert_new_object(cv::Mat &frame, TrackingBox &tbox)
{
    SimilarObj obj;
    obj.init(frame, tbox, uuid_);
    objects_.push_back(obj);
    obj_id_map_.emplace(tbox.id, uuid_);
    LOGW("new obj inserted id[%d-%d]  total:%d %d\n", tbox.id, uuid_,
        (int)objects_.size(), (int)obj_id_map_.size());
    uuid_++;
}

bool SimilarTracker::check_and_reinit(cv::Mat &frame, std::vector<TrackingBox> &tboxes)
{
    if (objects_.size() > 0) {
        no_obj_no_box_counter_ = 0;
        return false;
    }

    obj_id_map_.clear();
    if (tboxes.size() > 0) {
        for (auto &tbox : tboxes) {
            cur_frame_seq_ = tbox.frame;
            insert_new_object(frame, tbox);
        }
    } else {
        //not tracking object, not tracking box
        no_obj_no_box_counter_++;

        if (no_obj_no_box_counter_ >= SIM_TRK_ID_RESET_NUM) {
            uuid_ = 0;  //reset uuid
        }
    }
    return true;
}

std::vector<TrackingBox> SimilarTracker::getBoxes(void)
{
    std::vector<TrackingBox> result;
    for (const auto &obj : objects_) {
        int age = cur_frame_seq_ - obj.tbox_.frame;
        if (age <= SORT_OBJECT_HIDDEN_AGE_MAX) {
            TrackingBox tbox = obj.tbox_;
            if (tbox.id == main_trk_id_) {
                tbox.tracking = 1;
            } else {
                tbox.tracking = 0;
            }
            result.push_back(tbox);
        }
    }
    return result;
}

std::vector<TrackingBox> SimilarTracker::getHiddenBoxes(void)
{
    std::vector<TrackingBox> result;
    for (const auto &obj : objects_) {
        int age = cur_frame_seq_ - obj.tbox_.frame;
        if (age > SORT_OBJECT_HIDDEN_AGE_MAX) {
            TrackingBox tbox = obj.tbox_;
            if (tbox.id == main_trk_id_) {
                tbox.tracking = 1;
            } else {
                tbox.tracking = 0;
            }
            result.push_back(obj.tbox_);
        }
    }
    return result;
}

void SimilarTracker::find_safe_boxes(std::vector<int> &safe, std::vector<TrackingBox> &tboxes)
{
    if (tboxes.size() == 0) {
        return;
    } else if (tboxes.size() < 1) {
        safe.push_back(0);
        return;
    }
    std::vector<uint8_t> v_not_safe(tboxes.size(), 0);
    RectBox  box_a;
    RectBox  box_b;
    for (int i = 0; i < (int)tboxes.size(); i++) {
        if (v_not_safe[i]) {
            continue;
        }
        for (int j = i+1; j < (int)tboxes.size(); j++) {
            if (v_not_safe[j]) {
                continue;
            }
            if (tboxes[i].class_idx != tboxes[j].class_idx) {
                continue;   //not the same class, always safe
            }
#if 0
            //TODO:
            // - scale rect = rect*1.2; then GetIOU()
            // - face maybe ok, human not need
            #define RECT_SAFE_SCALE (1.1)
            box_a = tboxes[i].box;
            box_b = tboxes[j].box;
            box_a.width  *= RECT_SAFE_SCALE;
            box_a.height *= RECT_SAFE_SCALE;
            box_b.width  *= RECT_SAFE_SCALE;
            box_b.height *= RECT_SAFE_SCALE;
            float iou = GetIOU(box_a, box_b);
#else
            float iou = GetIOU(tboxes[i].box, tboxes[j].box);
#endif
            if (iou >= 0.001f) {
                v_not_safe[i] = v_not_safe[j] = 1;
            }
        }
    }
    for (int i = 0; i < (int)v_not_safe.size(); i++) {
        if (v_not_safe[i] == 0) {
            safe.push_back(i);
        }
    }
}

void SimilarTracker::check_and_remove_object(void)
{
    int removed = 0;
    auto it = objects_.begin();

    TrackingBox main_tbox;
    uint8_t     main_removed = 0;

    while (it != objects_.end()) {
        int age = cur_frame_seq_ - (*it).tbox_.frame;
        if (age >= SORT_OBJECT_AGE_MAX) {
            remove_sort_id((*it).sort_id_);
            LOGW("[remove_obj] id[%d-%d]\n", (*it).sort_id_, (*it).tbox_.id);

            if ((*it).tbox_.id == main_trk_id_) {
                main_removed = 1;
                main_tbox = (*it).tbox_;
            }

            it = objects_.erase(it);
            removed++;
        } else {
            it++;
        }
    }
    if (removed) {
        //LOGD("object removed, num:%d\n", removed);
    }

    if (main_removed) {
        //switch tracking target
        // - find closed of the same class
        // - do switch
        std::vector<TrackingBox> tboxes;
        for (const auto &obj : objects_) {
            if (obj.tbox_.class_idx == main_tbox.class_idx) {
                tboxes.push_back(obj.tbox_);
            }
        }
        if (tboxes.size() <= 0) {
            main_trk_id_ = -1;                //no match
        } else if (tboxes.size() == 1) {
            main_trk_id_ = tboxes[0].id;      //unique match
        } else {
            // find close
            int min_dist = 1000000000;
            int min_dist_obj_id = -1;
            for (auto& tbox : tboxes) {
                int dist = GetSimpleDist(tbox.box, main_tbox.box);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_obj_id = tbox.id;
                }
            }
            if (min_dist_obj_id >= 0) {
                main_trk_id_ = min_dist_obj_id;
            }
        }
        if (main_trk_id_ > 0) {
            LOGD("switch tracking:[%d]\n", main_trk_id_);
        }
    }
}

void SimilarTracker::remove_sort_id(int sort_id)
{
    auto it = obj_id_map_.find(sort_id);
    if (it != obj_id_map_.end()) {
        obj_id_map_.erase(it);
    }
}

#define BOX_USED    (-1)

std::vector<TrackingBox> SimilarTracker::Run(cv::Mat &frame, std::vector<TrackingBox> &tboxes)
{
    //debug_insert_20_box(tboxes);

    if (check_and_reinit(frame, tboxes)) {
        return getBoxes();
    }
    if (tboxes.size() <= 0) {
        cur_frame_seq_++;
        check_and_remove_object();
        return getBoxes();
    }

    cur_frame_seq_ = tboxes[0].frame;
    memset(curr_obj_cls_counter_, 0, sizeof(curr_obj_cls_counter_));

    //debug_check_box_valid(tboxes);

    std::vector<int> v_safe_box;
    find_safe_boxes(v_safe_box, tboxes);

    std::vector<int>     v_box_not_match;
    std::vector<uint8_t> v_obj_match(objects_.size(), 0);

    //full match
    for (int i = 0; i < (int) tboxes.size(); i++) {
        auto &tbox = tboxes[i];

        if (tboxes[i].class_idx < SORT_YOLO_CLASS_NUM) {
            curr_obj_cls_counter_[tboxes[i].class_idx]++;
        }

        int uid = -1;
        auto it = obj_id_map_.find(tbox.id);
        if (it == obj_id_map_.end()) {
            v_box_not_match.push_back(i);
            continue;
        }

        uid = it->second;
        int jj = 0;
        for (auto &obj: objects_) {
            if (obj.tbox_.id == uid) {
                //@Leo
                //TODO:
                // 1. when sort failed, sort_id maybe wrong
                // 2. may need a lot a calculation to compare the same object when object are too much
                // we need to track the main-tracking object Only
                //
                obj.updateBox(tbox);
                v_obj_match[jj] = 1;
                //LOGD("[full_match] update id:[%d-%d] match[%d]:%d\n", tbox.id, uid, jj, v_obj_match[jj]);
                break;
            }
            jj++;
        }
    }

    if (tboxes.size() > objects_.size()) {
        //new object
        for (auto i: v_box_not_match) {
            insert_new_object(frame, tboxes[i]);
        }
    } else {
        //compare unmatch object, with v_box_not_match
        for (int obj_idx = 0; obj_idx < (int) objects_.size(); obj_idx++) {
            if (v_obj_match[obj_idx]) {
                continue;
            }
            auto &obj = objects_[obj_idx];
            int box_not_match_idx = -1;
            int max_score_idx = -1;
            float max_score = -1;

            for (int kk = 0; kk < (int) v_box_not_match.size(); kk++) {
                int tbox_idx = v_box_not_match[kk];
                if (tbox_idx == BOX_USED) {
                    continue;   //used
                }
                auto &tbox = tboxes[tbox_idx];
                if (tbox.class_idx != obj.tbox_.class_idx) {
                    continue;
                }

                //TODO:
                // quich match only apply on close area()

                //quick match witn 1:1 match
                if (curr_obj_cls_counter_[tbox.class_idx] == 1 &&
                    last_obj_cls_counter_[tbox.class_idx] == 1) {
                    max_score_idx = tbox_idx;
                    max_score = 1.0f;
                    box_not_match_idx = kk;
                    LOGW("[re_match] solo quick match, id:[%d-%d]\n", tbox.id, obj.tbox_.id);
                    break;
                }
                float score = obj.checkMatchScore(frame, tbox.box);
                if (score > max_score) {
                    max_score = score;
                    max_score_idx = tbox_idx;
                    box_not_match_idx = kk;
                }
            }
            if (max_score_idx < 0) {
                continue; //not the same class
            }
            //TODO: @Leo
            // - only handle first object found
            // - use rect to find object match maybe better
            //
            //  object0: [score, score, score]
            //  object1: [score, score, score]
            if (max_score >= 0.3) {
                remove_sort_id(obj.sort_id_);
                obj_id_map_.emplace(obj.sort_id_, obj.tbox_.id);
                obj.sort_id_ = tboxes[max_score_idx].id;
                obj.updateBox(tboxes[max_score_idx]);
                v_box_not_match[box_not_match_idx] = BOX_USED;
                LOGW("[re_match] id:[%d-%d] OK, max score:%.2f\n", obj.tbox_.id, obj.tbox_.id, max_score);
            } else {
                LOGD("[re_match] id:[%d-%d] NG, max score:%.2f\n", obj.tbox_.id, obj.tbox_.id, max_score);
            }
        }
    }

    check_and_remove_object();

    //update object with safe iou
    if (v_safe_box.size() > 0) {
        //LOGD("v_safe_box:%d  obj_num:%d\n", (int) v_safe_box.size(), (int)objects_.size());
        for (auto idx: v_safe_box) {
            auto &tbox = tboxes[idx];
            auto it = obj_id_map_.find(tbox.id);
            if (it == obj_id_map_.end()) {
                continue;
            }
            //LOGD("[updat_roi] id:[%d-%d] >>\n", tbox.id, it->second);
            for (auto &obj: objects_) {
                if (obj.tbox_.id == it->second) {
                    //debug_image_similar(obj, frame, tbox);

                    //LOGD("[updat_roi] id:[%d-%d] done\n", tbox.id, obj.tbox_.id);
                    obj.update(frame, tbox);
                    break;
                }
            }
        }
    }

    memset(last_obj_cls_counter_, 0, sizeof(last_obj_cls_counter_));
    for (const auto &obj : objects_) {
        last_obj_cls_counter_[obj.tbox_.class_idx]++;
    }

    return getBoxes();
}

/*start tracking by ID/Rect */
void SimilarTracker::trackById(int id)
{
    LOGD("[start_tracking] id:%d\n", id);
    main_trk_id_ = id;
}

void SimilarTracker::trackByRect(const RectBox &rect)
{

}

void SimilarTracker::debug_image_similar(SimilarObj &obj, cv::Mat &frame, TrackingBox &tbox)
{

#if 0
    //if (obj.tbox_.class_idx == SORT_CLS_FACE) {
        RectBox box = tbox.box;
        LOGD("[DEBUG] value:%0.2f\n", obj.checkMatchScore(frame, box));

        box.x -= 10;
        box.x = MAX(0, box.x);
        LOGD("[DEBUG] [-10, 0] value:%0.2f\n", obj.checkMatchScore(frame, box));
        box.x -= 10;
        box.x = MAX(0, box.x);
        LOGD("[DEBUG] [-20, 0] value:%0.2f\n", obj.checkMatchScore(frame, box));
        box.x -= 10;
        box.x = MAX(0, box.x);
        LOGD("[DEBUG] [-30, 0] value:%0.2f\n", obj.checkMatchScore(frame, box));
        box.y -= 10;
        box.y = MAX(0, box.y);
        LOGD("[DEBUG] [0, -10] value:%0.2f\n", obj.checkMatchScore(frame, box));
        box.y -= 10;
        box.y = MAX(0, box.y);
        LOGD("[DEBUG] [0, -20] value:%0.2f\n", obj.checkMatchScore(frame, box));
        box.y -= 10;
        box.y = MAX(0, box.y);
        LOGD("[DEBUG] [0, -30] value:%0.2f\n", obj.checkMatchScore(frame, box));
    //}
#endif

}

void SimilarTracker::debug_check_box_valid(std::vector<TrackingBox> &tboxes)
{
    for (auto tbox:tboxes) {
        if (tbox.box.x >= 0 &&
            tbox.box.width >= 0 &&
            tbox.box.y >= 0 &&
            tbox.box.height >= 0 &&
            (tbox.box.x + tbox.box.width) <= 416 &&
            (tbox.box.y + tbox.box.height) <= 416) {
            continue;
        } else {
            LOGE("cls:%d id:%d box:%0.2f %0.2f %0.2f %0.2f\n",
                (int)tbox.class_idx, tbox.id,
                tbox.box.x,tbox.box.y,tbox.box.width,tbox.box.height);
        }
    }
}

void SimilarTracker::debug_insert_20_box(std::vector<TrackingBox> &tboxes)
{
    assert(frame.rows == 416);
    assert(frame.cols == 416);
    //debug insert for testing
    if (tboxes.size() > 0) {
        TrackingBox bbb = tboxes[0];
        for (int i = 0; i < 20; i++) {
            bbb.id++;
            bbb.class_idx = 0;
            bbb.box.x += 2;
            tboxes.push_back(bbb);
        }
        for (size_t j = 0; j < tboxes.size(); j++) {
            tboxes[j].class_idx = 0;
        }
    }
}
