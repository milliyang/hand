#pragma once

#include <stdio.h>
#include <string>
#include "opencv2/opencv.hpp"

using std::string;

class YuvReader
{
public:
	YuvReader();
	~YuvReader();

	void Init(const string &file, int width, int height);
	void Read(cv::Mat &mat);

private:
	int inited_;
	string movFile_;
	int width_;
	int height_;

	int err_;
	FILE *fin_;
	int size_;
	unsigned char *buffer_;
};

