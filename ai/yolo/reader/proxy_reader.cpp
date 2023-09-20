#include "proxy_reader.h"
#include <iostream>
#include <fstream>

ProxyReader::ProxyReader()
{
    stream_ = false;
    seq_ = 0;
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
    } else if (file.find(".txt") != std::string::npos) {
        source_ = SOURCE_TXT;
        stream_ = true;

        std::ifstream fin(file);
        std::string s;
        while (getline(fin,s)) {
            v_str_.push_back(s);
            //std::cout << "Read from file: " << s << std::endl;
        }
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
    } else if (source_ == SOURCE_TXT) {
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
    } else if (source_ == SOURCE_TXT) {
        if (seq_ >= v_str_.size()) {
            frame = cv::Mat();
            return;
        }
        cvCapture_.open(v_str_[seq_]);
        if (!cvCapture_.isOpened()) {
            frame = cv::Mat();
            return;
        }
        cvCapture_ >> frame;
        cvCapture_.release();
        seq_++;
    } else {
        cvCapture_ >> frame;
    }
}

void ProxyReader::close(void)
{
    imvt_zmq_sub_yuv_deinit();
    imvt_zmq_sub_deinit();

    if (source_ == SOURCE_TXT) {
        return;
    }

    // When everything done, release the video capture object
    cvCapture_.release();
}

bool ProxyReader::isStream(void)
{
    return stream_;
}

bool ProxyReader::isFilelist(void)
{
    return v_str_.size() > 0;
}

void ProxyReader::resetReadIndex(int index)
{
    if (isFilelist()) {
        if (index > 0) {
            int max_idx = (int)v_str_.size()-1;
            seq_ = std::max(index, 0);
            seq_ = std::min(index, max_idx);
        } else {
            seq_ -=2;
            seq_ = std::max(seq_, 0);
        }
    }
}

std::string ProxyReader::peekFile(void)
{
    if (seq_ < v_str_.size()) {
        return v_str_.at(seq_);
    } else {
        return std::string("");
    }
}
