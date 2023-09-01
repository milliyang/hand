#include "SimilarTracker.h"

#ifdef CONFIG_SPDLOG
#define LOG_TAG "SimTrker"
#include "log.h"
#endif

#define SSIM_FEATURE_SIZE   (8)

static bool check_box_valid(const cv::Rect_<float> &bb)
{
    if (bb.x >= 0 && bb.y >= 0) {
        return true;
    } else {
        //bb.x == nan
        return false;
    }
}

// Computes IOU between two bounding boxes
static float GetIOU(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
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
    cur_max_objects_ = 0;

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
    cur_max_objects_ = objects_.size();
    return true;
}

std::vector<TrackingBox> SimilarTracker::getBoxes(void)
{
    std::vector<TrackingBox> result;
    for (const auto &obj : objects_) {
        int age = cur_frame_seq_ - obj.tbox_.frame;
        if (age <= SORT_OBJECT_HIDDEN_AGE_MAX) {
            result.push_back(obj.tbox_);
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
            result.push_back(obj.tbox_);
        }
    }
    return result;
}

void SimilarTracker::find_safe_boxes(std::set<int> &safe, std::vector<TrackingBox> &tboxes)
{
    if (tboxes.size() == 0) {
        return;
    } else if (tboxes.size() < 1) {
        safe.insert(0);
        return;
    }
    std::vector<uint8_t> used(tboxes.size(), 0);
    for (int i = 0; i < (int)tboxes.size(); i++) {
        if (used[i]) {
            continue;
        }
        for (int j = i+1; j < (int)tboxes.size(); j++) {
            if (used[j]) {
                continue;
            }
            if (tboxes[i].class_idx != tboxes[j].class_idx) {
                continue;   //the same class, always safe
            }

            //TODO:
            // scale rect = rect*1.2; then GetIOU()
            //
            float iou = GetIOU(tboxes[i].box, tboxes[j].box);
            if (iou >= 0.001f) {
                used[i] = used[j] = 1;
            }
        }
    }
    for (int i = 0; i < used.size(); i++) {
        if (used[i] == 0) {
            safe.insert(i);
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
        cur_max_objects_ = (int)objects_.size();
        //LOGD("object removed, num:%d\n", removed);
    }
}

std::vector<TrackingBox> SimilarTracker::Run(cv::Mat &frame, std::vector<TrackingBox> &tboxes)
{
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

    std::set<int> safe_box_set;
    find_safe_boxes(safe_box_set, tboxes);

    std::vector<int> box_not_match;
    std::vector<uint8_t> obj_match(objects_.size(), 0);
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
            box_not_match.push_back(i);
            continue;
        }

        uid = it->second;
        //LOGD("00 id:%d -> uid:%d\n", tbox.id, uid);
        int obj_idx = 0;
        for (auto &obj: objects_) {
            if (obj.tbox_.id == uid) {
                obj.updateBox(tbox);
                obj_match[obj_idx] = 1;
                //LOGD("[full_match] update id:%d uid:%d match[%d]:%d\n", tbox.id, uid, obj_idx, obj_match[obj_idx]);
                break;
            }
            obj_idx++;
        }
    }

    //try new object
    if (tboxes.size() > cur_max_objects_) {
        LOGD("Try new object\n");
        for (auto i: box_not_match) {
            insert_new_object(frame, tboxes[i]);
        }
    } else {
        //compare unmatch object, with box_not_match
        for (int obj_idx = 0; obj_idx < objects_.size(); obj_idx++) {
            if (obj_match[obj_idx]) {
                continue;
            }

            auto &obj = objects_[obj_idx];
            float max_score_idx = -1;
            float max_score = -1;

            for (auto b_idx: box_not_match) {
                auto &tbox = tboxes[b_idx];
                if (tbox.class_idx != obj.tbox_.class_idx) {
                    continue;
                }

                //TODO:
                // quich match only apply on close area()

                //quick match witn 1:1 match
                if (obj_cls_counter[tbox.class_idx] == 1 &&
                    objects_cls_counter_[tbox.class_idx] == 1) {
                    max_score_idx = b_idx;
                    max_score = 1.0f;
                    LOGW("[re_match] solo quick match, id:[%d-%d]\n", obj.tbox_.id, obj.tbox_.id);
                    break;
                }
                float score = obj.checkMatchScore(frame, tbox.box);
                if (score > max_score) {
                    max_score = score;
                    max_score_idx = b_idx;
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
                LOGW("[re_match] id:[%d-%d] OK, max score:%.2f\n", obj.tbox_.id, obj.tbox_.id, max_score);
            } else {
                LOGD("[re_match] id:[%d-%d] NG, max score:%.2f\n", obj.tbox_.id, obj.tbox_.id, max_score);
            }
        }
    }

    check_and_remove_object();

    cur_max_objects_ = std::max(cur_max_objects_, (int)objects_.size());

    //update object safe
    if (safe_box_set.size() > 0) {
        //LOGD("safe_box_set:%d  obj_num:%d\n", (int) safe_box_set.size(), (int)objects_.size());
        for (auto idx: safe_box_set) {
            auto &tbox = tboxes[idx];
            int uid = -1;
            auto it = obj_id_map_.find(tbox.id);
            if (it == obj_id_map_.end()) {
                continue;
            }
            uid = it->second;
            //LOGD("[updat_roi] id:[%d-%d] >>\n", tbox.id, uid);
            for (auto &obj: objects_) {
                if (obj.tbox_.id == uid) {
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
void trackById(int id)
{

}

void trackByRect(const RectBox &rect)
{

}
