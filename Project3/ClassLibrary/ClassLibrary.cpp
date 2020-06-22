#include "pch.h"

#include "ClassLibrary.h"

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

#include <string.h>


using namespace cv;
using namespace std;
using namespace System;

using namespace System::Collections;

using namespace System::Windows::Forms;

using namespace System::Drawing;
using namespace cv;
using namespace std;
using namespace cv::ml;
using namespace System::Runtime::InteropServices;

cv::CascadeClassifier cascade;
cv::CascadeClassifier kytu;
cv::Ptr<SVM> SVM_Num;
cv::Ptr<SVM> SVM_Char;

const int number_of_feature_Number = 32;
const int number_of_feature_Character = 32;

cv::Mat ClassLibrary::XyLy::BitmapToMat(System::Drawing::Bitmap ^ bitmap)
{
	cv::Mat final_mat;

	System::Drawing::Imaging::BitmapData^ bmpdata = bitmap->LockBits(System::Drawing::Rectangle(0, 0, bitmap->Width, bitmap->Height),
		System::Drawing::Imaging::ImageLockMode::ReadWrite, System::Drawing::Imaging::PixelFormat::Format24bppRgb);

	if (bitmap->PixelFormat == System::Drawing::Imaging::PixelFormat::Format8bppIndexed)
	{
		final_mat = cv::Mat(cv::Size(bitmap->Width, bitmap->Height), CV_8UC1, bmpdata->Scan0.ToPointer(), bmpdata->Stride);
	}
	else
	{
		final_mat = cv::Mat(cv::Size(bitmap->Width, bitmap->Height), CV_8UC3, bmpdata->Scan0.ToPointer(), cv::Mat::AUTO_STEP);
		final_mat = cv::Mat(cv::Size(bitmap->Width, bitmap->Height), CV_8UC3, bmpdata->Scan0.ToPointer(), bmpdata->Stride);
	}

	bitmap->UnlockBits(bmpdata);

	return final_mat;
}




bool ClassLibrary::XyLy::Load_Cascade()
{
	try
	{
		cascade = cv::CascadeClassifier("C:\\Users\\LeHieu\\Desktop\\Project3\\Project3\\bienso.xml");
		kytu;// = cv::CascadeClassifier("kytu.xml");
		kytu.load("C:\\Users\\LeHieu\\Desktop\\Project3\\Project3\\kytu.xml");
		SVM_Num = StatModel::load<SVM>("C:\\Users\\LeHieu\\Desktop\\Project19\\Number.xml");
		SVM_Char = StatModel::load<SVM>("C:\\Users\\LeHieu\\Desktop\\Project19\\Char.xml");
		return true;
	}
	catch (const std::exception&)
	{
		return false;
	}
}



static int count_pixel(Mat img, bool black_pixel = true)
{
	int black = 0;
	int white = 0;
	for (int i = 0; i < img.rows; ++i)
		for (int j = 0; j < img.cols; ++j)
		{
			if (img.at<uchar>(i, j) == 0)
				black++;
			else
				white++;
		}
	if (black_pixel)
		return black;
	else
		return white;
}


static vector<float> calculate_feature(Mat src)
{
	Mat img;
	if (src.channels() == 3)
	{
		cvtColor(src, img, CV_BGR2GRAY);
		threshold(img, img, 100, 255, CV_THRESH_BINARY);
	}
	else
	{
		threshold(src, img, 100, 255, CV_THRESH_BINARY);
	}

	vector<float> r;
	//vector<int> cell_pixel;

	resize(img, img, cv::Size(40, 40));

	int h = img.rows / 4;
	int w = img.cols / 4;
	int S = count_pixel(img);
	int T = img.cols * img.rows;
	for (int i = 0; i < img.rows; i += h)
	{
		for (int j = 0; j < img.cols; j += w)
		{
			Mat cell = img(Rect(i, j, h, w));
			int s = count_pixel(cell);
			float f = (float)s / S;
			r.push_back(f);
		}
	}

	for (int i = 0; i < 16; i += 4)
	{
		float f = r[i] + r[i + 1] + r[i + 2] + r[i + 3];
		r.push_back(f);
	}

	for (int i = 0; i < 4; ++i)
	{
		float f = r[i] + r[i + 4] + r[i + 8] + r[i + 12];
		r.push_back(f);
	}

	r.push_back(r[0] + r[5] + r[10] + r[15]);
	r.push_back(r[3] + r[6] + r[9] + r[12]);
	r.push_back(r[0] + r[1] + r[4] + r[5]);
	r.push_back(r[2] + r[3] + r[6] + r[7]);
	r.push_back(r[8] + r[9] + r[12] + r[13]);
	r.push_back(r[10] + r[11] + r[14] + r[15]);
	r.push_back(r[5] + r[6] + r[9] + r[10]);
	r.push_back(r[0] + r[1] + r[2] + r[3] + r[4] + r[7] + r[8] + r[11] + r[12] + r[13] + r[14] + r[15]);

	return r; //32 feature
}

char character_Number(Mat img_character)
{
	char c = '*';

	vector<float> feature = calculate_feature(img_character);
	// Open CV3.1
	Mat m = Mat(1, number_of_feature_Number, CV_32FC1);
	for (size_t i = 0; i < feature.size(); ++i)
	{
		float temp = feature[i];
		m.at<float>(0, i) = temp;
	}

	int ri = int(SVM_Num->predict(m)); // 
	/*int ri = int(svmNew.predict(m));*/
	if (ri >= 0 && ri <= 9)
		c = (char)(ri + 48); //ma ascii 0 = 48
	//if (ri >= 10 && ri < 18)
	//	c = (char)(ri + 55); //ma accii A = 5, --> tu A-H
	//if (ri >= 18 && ri < 22)
	//	c = (char)(ri + 55 + 2); //K-N, bo I,J
	//if (ri == 22) c = 'P';
	//if (ri == 23) c = 'S';
	//if (ri >= 24 && ri < 27)
	//	c = (char)(ri + 60); //T-V,  
	//if (ri >= 27 && ri < 30)
	//	c = (char)(ri + 61); //X-Z
	//if (ri == 30) c = 'W';
	//if (ri == 31) c = 'O';
	//if (ri == 32) c = 'I';
	return c;
}

char PredictSVMNumber(Mat Number)
{
	vector<Mat> draw_character;
	Mat plateImg, chaImg;

	draw_character.push_back(Number);

	convertScaleAbs(Number, plateImg);
	convertScaleAbs(Number, chaImg);
	//imshow("A", Number);
	char cs = character_Number(Number);
	return cs;
}

char character_Chars(Mat img_character)
{
	char c = '*';

	vector<float> feature = calculate_feature(img_character);
	// Open CV3.1
	Mat m = Mat(1, number_of_feature_Character, CV_32FC1);
	for (size_t i = 0; i < feature.size(); ++i)
	{
		float temp = feature[i];
		m.at<float>(0, i) = temp;
	}

	int ri = int(SVM_Char->predict(m)); // 
	ri = ri + 10;
	/*int ri = int(svmNew.predict(m));*/
	//if (ri >= 0 && ri <= 9)
	//	c = (char)(ri + 48); //ma ascii 0 = 48
	if (ri >= 10 && ri < 18)
		c = (char)(ri + 55); //ma accii A = 5, --> tu A-H
	if (ri >= 18 && ri < 22)
		c = (char)(ri + 55 + 2); //K-N, bo I,J
	if (ri == 22) c = 'P';
	if (ri == 23) c = 'S';
	if (ri >= 24 && ri < 27)
		c = (char)(ri + 60); //T-V,  
	if (ri >= 27 && ri < 30)
		c = (char)(ri + 61); //X-Z
	if (ri == 30) c = 'W';
	if (ri == 31) c = 'O';
	if (ri == 32) c = 'I';
	return c;

}

char PredictSVMChar(Mat Kytu)
{
	vector<Mat> draw_character;
	Mat plateImg, chaImg;
	draw_character.push_back(Kytu);
	convertScaleAbs(Kytu, plateImg);
	convertScaleAbs(Kytu, chaImg);
	char cs = character_Chars(Kytu);
	return cs;
}


System::String ^ Convert_CharToString(char giatri)
{
	switch (giatri)
	{
		case '0': return "0"; 
		case '1': return "1";
		case '2': return "2";
		case '3': return "3";
		case '4': return "4";
		case '5': return "5";
		case '6': return "6";
		case '7': return "7";
		case '8': return "8";
		case '9': return "9";
		//- Kí tự trên biển số A, B, C, D, E, F, G, H, K, L, M, N, P, S, T, U, V, X, Y, Z
		case 'A': return "A";
		case 'B': return "B";
		case 'C': return "C";
		case 'D': return "D";
		case 'E': return "E";
		case 'F': return "F";
		case 'G': return "G";
		case 'H': return "H";
		case 'K': return "K";
		case 'L': return "L";
		case 'M': return "M";
		case 'N': return "N";
		case 'P': return "P";
		case 'S': return "S";
		case 'T': return "T";
		case 'U': return "U";
		case 'V': return "V";
		case 'X': return "X";
		case 'Y': return "Y";
		case 'Z': return "Z";
	}
}



System::String ^ ClassLibrary::XyLy::DocDuLieu(cv::Mat HinhAnh)
{
	System::String^ GiaTri="";
	Mat anh;
	Mat bx_cat;
	Mat img_color;
	Mat face_roi;
	cv::Mat matGray;

	//cv::Mat matGray = imread("a21.jpg", IMREAD_GRAYSCALE);// sửa lại ảnh---------------------------------------------

	cvtColor(HinhAnh, matGray, CV_BGR2GRAY);


	//detect
	std::vector<cv::Rect> rects;

	// tìm kiếm
	cascade.detectMultiScale(matGray, rects, 1.1, 3, CV_HAAR_SCALE_IMAGE);
	img_color = matGray;
	for (int n = 0; n < rects.size(); n++) {
		//rectangle(img_color, rects[n], cv::Scalar(255, 0, 0), 2);
		matGray(rects[n]).copyTo(face_roi);
	}
	if (rects.size() == 0)
	{
		waitKey(0);
		return "";
	}
	cv::resize(face_roi, bx_cat, cv::Size(408, 300));
	cv::blur(bx_cat, bx_cat, cv::Size(3, 3), cv::Point(0, 0), 0);
	adaptiveThreshold(bx_cat, bx_cat, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 35, 6);
	std::vector<Rect> r_kt;
	kytu.detectMultiScale(bx_cat, r_kt, 1.1, 2, 0, cv::Size(35, 40));
	for (int i = 0; i < r_kt.size() - 1; i++)
	{
		for (int j = i + 1; j < r_kt.size(); j++)
		{
			if (r_kt[i].y < 120 && r_kt[j].y > 120)
			{
				Rect tamp;
				tamp = r_kt[i];
				r_kt[i] = r_kt[j];
				r_kt[j] = tamp;
			}

			if (r_kt[i].y > 120 && r_kt[j].y < 120)
			{
				Rect tamp;
				tamp = r_kt[i];
				r_kt[i] = r_kt[j];
				r_kt[j] = tamp;
			}

			if (r_kt[i].y < 120 && r_kt[j].y < 120 && r_kt[i].x > r_kt[j].x)
			{
				Rect tamp;
				tamp = r_kt[i];
				r_kt[i] = r_kt[j];
				r_kt[j] = tamp;

			}
			if (r_kt[i].y > 120 && r_kt[j].y > 120 && r_kt[i].x > r_kt[j].x)
			{
				Rect tamp;
				tamp = r_kt[i];
				r_kt[i] = r_kt[j];
				r_kt[j] = tamp;

			}
		}
	}
	for (int i = 0; i < r_kt.size(); i++)
	{
		if (r_kt[i].y >= 60 && r_kt[i].y <= 130)
			r_kt.erase(r_kt.begin() + i);
	}

	for (int n = 0; n < r_kt.size(); n++)
	{
		//rectangle(bx_cat, r_kt[n], cv::Scalar(0, 255, 0), 2);

		int x, y;
		x = r_kt[n].x;
		y = r_kt[n].y;

		if (y < 100 && (x > 204 && x < 250))
		{
			Mat tachChuoi;
			bx_cat(r_kt[n]).copyTo(tachChuoi);
			char cs = PredictSVMChar(tachChuoi);
			GiaTri += Convert_CharToString(cs);
		}
		else
		{
			Mat tachChuoi;
			bx_cat(r_kt[n]).copyTo(tachChuoi);
			char cs = PredictSVMNumber(tachChuoi);
			GiaTri += Convert_CharToString(cs);
		}
	}
	return GiaTri;
}



//System::String ^ ClassLibrary::XyLy::XuLyDuLieu(System::Drawing::Bitmap ^ bitmap)
//{
//	/*Mat Anh_BS = BitmapToMat(bitmap);
//		
//	System::String ^ GiaTri = DocDuLieu(Anh_BS);
//	
//		return GiaTri;*/
//}

System::String ^ ClassLibrary::XyLy::test2()
{
	// TODO: insert return statement here
	System::String^ xc = "hello";
	return xc;
}



System::String ^ ClassLibrary::XyLy::XuLyDuLieu(System::Drawing::Bitmap^ bitmap)
{
	//cv::Mat matGray = imread("C:\\Users\\LeHieu\\source\\repos\\Project3\\Project3\\a21.jpg", IMREAD_GRAYSCALE);

	//System::Drawing::Bitmap ^ cx;
	//cx->FromFile("C:\\Users\\LeHieu\\source\\repos\\Project3\\Project3\\a21.jpg");

	cv::Mat Anh_BS = BitmapToMat(bitmap);
	//cv::imshow("cx", Anh_BS);

	System::String ^ GiaTri = DocDuLieu(Anh_BS);
	return GiaTri;
}


//string ClassLibrary::XyLy::XuLyDuLieu(System::Drawing::Bitmap ^ bitmap)
//{
//	Mat Anh_BS = BitmapToMat(bitmap);
//	
//	string GiaTri = DocDuLieu(Anh_BS);
//
//	return GiaTri;
//}



