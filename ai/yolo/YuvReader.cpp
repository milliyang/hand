//#include "stdafx.h"
#include "YuvReader.h"

using std::string;

//#include <cv.h>
#include <opencv2/imgproc/types_c.h>	//bugfix:u20.04 opencv4 set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")   #u20.04
#include "opencv2/opencv.hpp"

YuvReader::YuvReader(void)
{
	inited_ = 0;
}

void YuvReader::Init(const string &file, int width, int height)
{
	movFile_ = file;
	width_ = width;
	height_ = height;

	//#err_ = fopen_s(&fin_, file.c_str(), "rb");	//win32
	// if (err_ != 0) {
	// 	std::cerr << "error: unable to open file :" << file << std::endl;
	// }
	fin_ = fopen(file.c_str(), "rb");
	if (fin_ == nullptr) {
		std::cerr << "error: unable to open file :" << file << std::endl;	
	}

	size_ = width * height * 3 / 2; // w*h*1.5
	buffer_ = new unsigned char[size_];

	inited_ = 1;
	std::cout << "inited done" << std::endl;
}

void YuvReader::Read(cv::Mat &mat)
{
	int len;
	//check last result
	if (err_ != 0) {
		mat.release();
		return;
	}

	len = fread(buffer_, size_ * sizeof(unsigned char), 1, fin_);
	if (len != 1) {
		err_ = -1;
		mat.release();
		return;
	}

	cv::Mat yuvImg;
	yuvImg.create(height_ * 3 / 2, width_, CV_8UC1);
	memcpy(yuvImg.data, buffer_, size_ * sizeof(unsigned char));
	cv::cvtColor(yuvImg, mat, CV_YUV2BGR_NV21);
	//cv::imshow("YxxxxxoTracker", mat);
	return;
}

YuvReader::~YuvReader()
{
	if (inited_) {
		fclose(fin_);
	}
}
