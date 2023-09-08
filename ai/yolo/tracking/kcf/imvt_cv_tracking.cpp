// #include "stdafx.h"

#include "imvt_cv_tracking.h"

#define LOGD   printf
#define LOGW   printf

#include <string>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>


using namespace std;

struct trk_context {
    cv::Ptr<cv::Tracker> tracker;
    cv::TrackerKCF::Params param;
    short first_frame;
    cv::Rect2d bbox;
    cv::Rect2d bbox_init;
};

static struct trk_context ctx;

void imvt_cv_tracking_enable(int enable, int x, int y, int width, int height)
{
    ctx.first_frame = 1;
    ctx.bbox_init.x = x;
    ctx.bbox_init.y = y;
    ctx.bbox_init.width = width;
    ctx.bbox_init.height = height;

    LOGD("%s init tracking x:%d y:%d w:%d h:%d\n", __FUNCTION__, x, y, width, height);
}

void debug_dump_mat(cv::Mat &mat)
{
    LOGD("%s\n", __FUNCTION__);
    FILE *fp = fopen("yolo.yuv", "w+");
    // fwrite(mat.data, img_size * 3 / 2, 1, fp);
    fwrite(mat.data, mat.size().area() * 3 / 2, 1, fp);
    fflush(fp);
    fclose(fp);
    LOGD("%s yuv done\n", __FUNCTION__);
}

void debug_read_mat(cv::Mat &mat, int img_size)
{
    FILE *fp = fopen("yolo.yuv", "rb");
    int r = img_size * 3 / 2;
    int len = fread(mat.data, r, 1, fp);
    LOGD("len:%d\n", len);
    fclose(fp);
}

int imvt_cv_tracking_detect(cv::Mat &mat, cv::Rect2d &box)
{
    bool ok = true;

    if (ctx.first_frame) {
        ctx.tracker = cv::TrackerKCF::create(ctx.param);
        ctx.tracker->init(mat, ctx.bbox_init);
        ctx.bbox = ctx.bbox_init;
        ctx.first_frame = 0;
#if 0
        debug_dump_mat(mat);
#endif
    } else {
        cv::Rect2i bbox; //fix for #4.5.5
        bool ok = ctx.tracker->update(mat, bbox);
        if (ok) {
            box = bbox;
            LOGD("bbox:%.2f %.2f %.2f %.2f\n", ctx.bbox.x, ctx.bbox.y, ctx.bbox.width, ctx.bbox.height);
        } else {
            LOGW("bbox:ng\n");
        }
    }
    return ok;
}

void imvt_cv_tracking_init(void)
{
    ctx.first_frame = 1;
    ctx.bbox_init.x = 1;
    ctx.bbox_init.y = 1;
    ctx.bbox_init.width = 100;
    ctx.bbox_init.height = 100;

    //setup KCF tracking param
    ctx.param.detect_thresh = 0.5f;
    ctx.param.sigma = 0.2f;
    ctx.param.lambda = 0.0001f;
    ctx.param.interp_factor = 0.075f;
    ctx.param.output_sigma_factor = 1.0f / 16.0f;
    ctx.param.resize = true;
    ctx.param.max_patch_size = 80 * 80;
    ctx.param.split_coeff = true;
    ctx.param.wrap_kernel = false;

#if 1
    ctx.param.desc_npca = cv::TrackerKCF::GRAY;
    ctx.param.desc_pca = cv::TrackerKCF::CN;
#else
    ctx.param.desc_npca = 0;
    ctx.param.desc_pca = cv::TrackerKCF::GRAY | cv::TrackerKCF::CN;
#endif
    //feature compression
    ctx.param.compress_feature = true;
    ctx.param.compressed_size = 2;
    ctx.param.pca_learning_rate = 0.15f;
}

void imvt_cv_tracking_deinit(void)
{

}