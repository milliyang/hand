#ifndef __IMVT_CV_TRACKING_H__
#define __IMVT_CV_TRACKING_H__

#include "opencv2/opencv.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/core/ocl.hpp"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* __cplusplus */

void imvt_cv_tracking_enable(int enable, int x, int y, int width, int height);
int  imvt_cv_tracking_detect(cv::Mat &mat, cv::Rect2d &box);
void imvt_cv_tracking_init(void);
void imvt_cv_tracking_deinit(void);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */

#endif /* __IMVT_CV_TRACKING_H__ */
