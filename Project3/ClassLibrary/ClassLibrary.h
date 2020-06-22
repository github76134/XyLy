#pragma once



#include "opencv2\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>

#include <opencv2/opencv.hpp>
#include <vector>
#include <assert.h>


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

using namespace System;
using namespace std;

namespace ClassLibrary {
	public ref class XyLy
	{
	public:
		cv::Mat BitmapToMat(System::Drawing::Bitmap^ bitmap);
	public:
		bool Load_Cascade();
	public:
		System::String^ XuLyDuLieu(System::Drawing::Bitmap ^ bitmap);
	public:
		System::String^ test2();
	public:
		System::String ^ DocDuLieu(cv::Mat HinhAnh);
	};
}
