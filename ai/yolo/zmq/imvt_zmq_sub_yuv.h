#ifndef __IMVT_ZMQ_SUB_YUV_H__
#define __IMVT_ZMQ_SUB_YUV_H__

//#include <cv.h>
#include <opencv2/opencv.hpp>

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* __cplusplus */

#ifndef OK
#define OK (0)
#define NG (-1)
#endif

void imvt_zmq_sub_yuv_init(int width, int height);
void imvt_zmq_sub_yuv_recv(cv::Mat &mat);
void imvt_zmq_sub_yuv_deinit(void);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* __cplusplus */
#endif /* __IMVT_ZMQ_SUB_YUV_H__ */
