#include "proxy_reader.h"

ProxyReader::ProxyReader()
{
    stream_ = false;
}

bool ProxyReader::open(std::string &file)
{
    // Create a VideoCapture object and open the input file
    // If the input is the web camera, pass 0 instead of the video file name
    //cv::VideoCapture cap(0); // open the default camera

    if (file.find(".jpg") != std::string::npos) {
        source_ = SOURCE_JPG;
    } else if (file.find(".png") != std::string::npos) {
        source_ = SOURCE_JPG;
    } else if (file.find(".bmp") != std::string::npos) {
        source_ = SOURCE_JPG;
    } else if (file.find(".yuv") != std::string::npos) {
        source_ = SOURCE_YUV;
        stream_ = true;
    } else if (file.find("tcp") != std::string::npos) {
        //"tcp://192.168.9.6:5556"
        source_ = SOURCE_NET;
        stream_ = true;
        std::cerr << "addr:" << file << std::endl;
    } else {
        source_ = SOURCE_MOV;
        stream_ = true;
    }

    if (source_ == SOURCE_YUV) {
        yuvReader_.Init(file, 416, 416);
        //yuvReader_.Init(file, 352, 288);
    } else if (source_ == SOURCE_NET) {
        imvt_zmq_sub_init((char*) file.c_str());
        imvt_zmq_sub_yuv_init(416, 416);
    } else {
        cvCapture_.open(file);
        // Check if camera opened successfully
        if (!cvCapture_.isOpened()) {
            std::cout << "Error opening video stream or file" << std::endl;
            return false;
        }
    }
    return true;
}

void ProxyReader::read(cv::Mat &frame)
{
    // Capture frame-by-frame
    if (source_ == SOURCE_YUV) {
        yuvReader_.Read(frame);
    } else if (source_ == SOURCE_NET) {
        imvt_zmq_sub_yuv_recv(frame);
    } else {
        cvCapture_ >> frame;
    }
}

void ProxyReader::close(void)
{
    imvt_zmq_sub_yuv_deinit();
    imvt_zmq_sub_deinit();

    // When everything done, release the video capture object
    cvCapture_.release();
}

bool ProxyReader::isStream(void)
{
    return stream_;
}