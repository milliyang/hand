#pragma once
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#include "YuvReader.h"
#include "imvt_zmq_sub.h"
#include "imvt_zmq_sub_yuv.h"

class ProxyReader
{

    enum SOURCE {
        SOURCE_JPG = (0),
        SOURCE_MOV = (1),
        SOURCE_YUV = (2),
        SOURCE_NET = (3),
        SOURCE_TXT = (4),   //filelist.txt
    };

public:
    ProxyReader();

    /*
     * support type:
     *       one image: jpg/png/bmp
     *          video : yuv/mov
     *  network stream: tcp://192.168.9.6:5556  //zmq publisher
     */
    bool open(std::string &file);
    void read(cv::Mat &frame);
    void close();

    bool isStream(void);

    /*filelist only*/
    bool isFilelist(void);
    void resetReadIndex(int index);
    std::string peekFile(void);
    /*filelist only*/

private:

    YuvReader yuvReader_;
    cv::VideoCapture cvCapture_;
    int source_;
    bool stream_;

    std::vector<std::string> v_str_;
    int seq_;
};
