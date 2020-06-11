#include <stdlib.h>
#include <stdio.h>
#include "convLayer.h"

#include "hls_opencv.h"
#include <opencv2/opencv.hpp>

#define MAX_HEIGHT 1080
#define MAX_WIDTH 1920

void simpleTest();
using namespace cv;

int main(int argc, char** argv) {
	Mat_<float> img(MAX_HEIGHT, MAX_WIDTH, CV_32FC1);
	Mat_<float> result(MAX_HEIGHT, MAX_WIDTH, CV_32FC1);
	Mat img2;

	img = imread("E:\\Fac\\SoC\\Projet\\images\\baseGray.jpg", 0);

	img.convertTo(img2, CV_32FC1);


	float filter[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	volatile float * srcArray = img2.ptr<float>(0);
	volatile float * resultPtr = result.ptr<float>(0);


	convLayer(srcArray, resultPtr, filter, MAX_HEIGHT, MAX_WIDTH);

	Mat resFinal;
	result.convertTo(resFinal, CV_8UC1);

	imwrite("E:\\Fac\\SoC\\Projet\\images\\convMasterV2.jpg", resFinal);

}

void simpleTest() {
	float data[6 * 4] = { 7, 5, 6, 3, 8, 2, 9, 1, 2, 4, 0, 9, 2, 4, 3, 1, 9, 5,
			5, 6, 0, 2, 6, 2 };
	float filter[9] = { 0 };
	/*for (int i = 0; i < 9; ++i) {
	 filter[i] =
	 }*/
	filter[4] = 1;

	float outputData[4 * 2] = { 0 };

	convMasterV2(data, outputData, filter, 4, 6);

	for (int i = 0; i < 4 * 2; i++) {
		printf("%d : %f\r\n", i, outputData[i]);
	}

}
