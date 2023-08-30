//#include "stdafx.h"

#include <opencv2/imgproc/types_c.h>	//bugfix:u20.04 opencv4 set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")   #u20.04
#include "opencv2/opencv.hpp"
#include "imvt_zmq_sub.h"
#include "imvt_zmq_sub_yuv.h"

struct imvt_mq_ctx {
    int width;
    int height;
    int size;
    unsigned char *buffer;
};
static struct imvt_mq_ctx s_zmq;

void imvt_zmq_sub_yuv_init(int width, int height)
{
    memset(&s_zmq, 0, sizeof(s_zmq));
    s_zmq.width = width;
    s_zmq.height = height;
    s_zmq.size = width * height * 3 / 2;
    s_zmq.buffer = new unsigned char[s_zmq.size];
}

void imvt_zmq_sub_yuv_recv(cv::Mat &mat)
{
    int len = imvt_zmq_sub_recv_frame(s_zmq.buffer, s_zmq.size);
    if (len != s_zmq.size) {
        mat.release();
    }

    cv::Mat yuvImg;
    yuvImg.create(s_zmq.height * 3 / 2, s_zmq.width, CV_8UC1);
    memcpy(yuvImg.data, s_zmq.buffer, s_zmq.size * sizeof(unsigned char));
    cv::cvtColor(yuvImg, mat, CV_YUV2BGR_NV21);
}

void imvt_zmq_sub_yuv_deinit(void)
{
    if (s_zmq.buffer != NULL) {
        delete s_zmq.buffer;
    }
}