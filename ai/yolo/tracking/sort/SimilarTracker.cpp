#include "SimilarTracker.h"

#define LOG_TAG "SimTrker"

#ifdef CONFIG_SPDLOG
#include "log.h"
#else
#include "sys/log/imvt_log.h"
#endif

#define SSIM_FEATURE_SIZE   (8)

static bool check_box_valid(const RectBox &bb)
{
    if (bb.x >= 0 && bb.y >= 0) {
        return true;
    } else {
        //bb.x == nan
        return false;
    }
}

// Computes IOU between two bounding boxes
static float GetIOU(RectBox bb_test, RectBox bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;
    if (un < DBL_EPSILON) {
        return 0;
    }
    return (float)(in / un);
}

SimilarTracker::SimilarTracker(void)
{
    uuid_ = 1;
    main_trk_id_ = -1;

    memset(objects_cls_counter_, 0, sizeof(objects_cls_counter_));
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
    LOGD("new obj inserted id[%d-%d]\n", tbox.id, uuid_);
    uuid_++;
}

bool SimilarTracker::check_and_reinit(cv::Mat &frame, std::vector<TrackingBox> &tboxes)
{
    if (objects_.size() > 0) {
        return false;
    }

    obj_id_map_.clear();
    if (tboxes.size() > 0) {
        for (auto &tbox : tboxes) {
            cur_frame_seq_ = tbox.frame;
            insert_new_object(frame, tbox);
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
    std::vector<uint8_t> v_used(tboxes.size(), 0);
    for (int i = 0; i < (int)tboxes.size(); i++) {
        if (v_used[i]) {
            continue;
        }
        for (int j = i+1; j < (int)tboxes.size(); j++) {
            if (v_used[j]) {
                continue;
            }
            if (tboxes[i].class_idx != tboxes[j].class_idx) {
                continue;   //not the same class, always safe
            }

            //TODO:
            // scale rect = rect*1.2; then GetIOU()
            //
            float iou = GetIOU(tboxes[i].box, tboxes[j].box);
            if (iou >= 0.001f) {
                v_used[i] = v_used[j] = 1;
            }
        }
    }
    for (int i = 0; i < (int)v_used.size(); i++) {
        if (v_used[i] == 0) {
            safe.push_back(i);
        }
    }
}

void SimilarTracker::check_and_remove_object(void)
{
    int removed = 0;
    auto it = objects_.begin();
    while (it != objects_.end()) {
        int age = cur_frame_seq_ - (*it).tbox_.frame;
        if (age >= SORT_OBJECT_AGE_MAX) {
            auto it_id = obj_id_map_.find((*it).sort_id_);
            if (it_id != obj_id_map_.end()) {
                obj_id_map_.erase(it_id);
            }
            LOGD("[remove_obj] id[%d-%d]\n", (*it).sort_id_, (*it).tbox_.id);
            it = objects_.erase(it);
            removed++;
        } else {
            it++;
        }
    }
    if (removed) {
        //LOGD("object removed, num:%d\n", removed);
    }
}

#define BOX_USED    (-1)

std::vector<TrackingBox> SimilarTracker::Run(cv::Mat &frame, std::vector<TrackingBox> &tboxes)
{
    assert(frame.rows == 416);
    assert(frame.cols == 416);
    if (check_and_reinit(frame, tboxes)) {
        return getBoxes();
    }
    if (tboxes.size() <= 0) {
        cur_frame_seq_++;
        check_and_remove_object();
        return getBoxes();
    } else {
        cur_frame_seq_ = tboxes[0].frame;
    }

    std::vector<int> v_safe_box;
    find_safe_boxes(v_safe_box, tboxes);

    std::vector<int>     v_box_not_match;
    std::vector<uint8_t> v_obj_match(objects_.size(), 0);
    uint8_t obj_cls_counter[SORT_YOLO_CLASS_NUM];
    memset(obj_cls_counter, 0, sizeof(obj_cls_counter));

    //full match
    for (int i = 0; i < (int) tboxes.size(); i++) {
        auto &tbox = tboxes[i];

        if (tboxes[i].class_idx < SORT_YOLO_CLASS_NUM) {
            obj_cls_counter[tboxes[i].class_idx]++;
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

            for (int kk = 0; kk < v_box_not_match.size(); kk++) {
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
                if (obj_cls_counter[tbox.class_idx] == 1 &&
                    objects_cls_counter_[tbox.class_idx] == 1) {
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
            //TODO:
            // Leo: only hand one object now, first found
            //
            //  object0: [score, score, score]
            //  object1: [score, score, score]
            if (max_score >= 0.3) {
                obj.updateBox(tboxes[max_score_idx]);
                obj.sort_id_ = tboxes[max_score_idx].id;
                obj_id_map_.emplace(obj.sort_id_, obj.tbox_.id);
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
                    //debug_image_ssim(obj, frame, tbox);

                    //LOGD("[updat_roi] id:[%d-%d] done\n", tbox.id, uid);
                    obj.update(frame, tbox);
                    break;
                }
            }
        }
    }

    memset(objects_cls_counter_, 0, sizeof(objects_cls_counter_));
    for (const auto &obj : objects_) {
        objects_cls_counter_[obj.tbox_.class_idx] ++;
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

void SimilarTracker::debug_image_ssim(SimilarObj &obj, cv::Mat &frame, TrackingBox &tbox)
{

#if 0
    if (obj.tbox_.class_idx == SORT_CLS_FACE) {
        RectBox box = tbox.box;
        LOGD("[DEBUG] SSIM:%0.2f\n", obj.checkMatchScore(frame, box));

        box.x -= 10;
        box.x = MAX(0, box.x);
        LOGD("[DEBUG] [-10, 0] SSIM:%0.2f\n", obj.checkMatchScore(frame, box));
        box.x -= 10;
        box.x = MAX(0, box.x);
        LOGD("[DEBUG] [-20, 0] SSIM:%0.2f\n", obj.checkMatchScore(frame, box));
        box.x -= 10;
        box.x = MAX(0, box.x);
        LOGD("[DEBUG] [-30, 0] SSIM:%0.2f\n", obj.checkMatchScore(frame, box));
        box.y -= 10;
        box.y = MAX(0, box.y);
        LOGD("[DEBUG] [0, -10] SSIM:%0.2f\n", obj.checkMatchScore(frame, box));
        box.y -= 10;
        box.y = MAX(0, box.y);
        LOGD("[DEBUG] [0, -20] SSIM:%0.2f\n", obj.checkMatchScore(frame, box));
        box.y -= 10;
        box.y = MAX(0, box.y);
        LOGD("[DEBUG] [0, -30] SSIM:%0.2f\n", obj.checkMatchScore(frame, box));
    }
#endif

}
