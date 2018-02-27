#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>   

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <cmath>
using namespace dlib;
using namespace std;
using namespace cv;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;

std::vector<cv::Point> setLandmarkPoints(int a, int b, full_object_detection& shape)
{
	std::vector<cv::Point> res;
	cv::Point pt = cv::Point(0, 0);
	for (int i = a; i < b; i++)
	{
		auto j = shape.part(i);
		//cout << j.x() << endl;
		pt.x = j.x();
		pt.y = j.y();
		res.push_back(pt);
	}
	return res;
}

double getDistance(cv::Point pointO, cv::Point pointA)
{
	double distance;
	distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
	distance = sqrtf(distance);

	return distance;
}

double cal_percentage(double a, double b, double c)
{
	double res;
	res = (a + b) / (2 * c);
	return res;
}

int main()
{
	int camera = 0;
	int flag = 0; // whether valid user
	string source = "admin_data/admin.jpg";
	const static Scalar colors[] =
	{
		Scalar(0,0,255),
		Scalar(255,255,255)
	};

	VideoCapture capture;
	Mat frame,gray;
	cv::Rect alternative;
	
	frontal_face_detector detector = get_frontal_face_detector();

	shape_predictor sp;
	deserialize("shape_predictor_5_face_landmarks.dat") >> sp;

	anet_type net;
	deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

	shape_predictor sp2;
	deserialize("shape_predictor_68_face_landmarks.dat") >> sp2;

	// read admin image
	std::vector<matrix<rgb_pixel>> admin_imgs;
	matrix<rgb_pixel> admin_img;
	load_image(admin_img, source);
	admin_imgs.push_back(admin_img);
	std::vector<matrix<float, 0, 1>> admin_descriptors = net(admin_imgs);

	if (!capture.open(camera))
	{ 
		cout << "Capture from camera #" << camera << " didn't work" << endl;
		system("pause");
		return 0;
	}

	for (;;)
	{
		capture >> frame;
		//resize(frame, frame, Size(), 500 / frame.cols, 500 / frame.cols);
		cv_image<bgr_pixel> cimg(frame);

		std::vector<matrix<rgb_pixel>> faces;
		if (flag == 0)
		{
			for (auto face : detector(cimg))
			{
				auto shape = sp(cimg, face);
				matrix<rgb_pixel> face_chip;
				extract_image_chip(cimg, get_face_chip_details(shape, 150, 0.25), face_chip);
				faces.push_back(move(face_chip));

				alternative.x = face.left();
				alternative.y = face.top();
				alternative.width = face.right() - face.left();
				alternative.height = face.bottom() - face.top();
				cv::rectangle(frame, alternative, colors[0],2);
			}

			if (faces.size() != 0)
			{
				std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
				for (int i = 0; i < face_descriptors.size(); i++)
				{
					if (length(admin_descriptors[0] - face_descriptors[i]) < 0.4)
					{
						flag = 1;
						cout << " admin detected !!" << endl;
						break;
					}
				}
			}
		}
		else
		{
			for (auto face : detector(cimg))
			{
				alternative.x = face.left();
				alternative.y = face.top();
				alternative.width = face.right() - face.left();
				alternative.height = face.bottom() - face.top();
				cv::rectangle(frame, alternative, colors[1],2);

				// eyes detect and draw
				full_object_detection shape2 = sp2(cimg, face);
				std::vector<cv::Point> lefteye = setLandmarkPoints(36,42,shape2);
				std::vector<cv::Point> righteye = setLandmarkPoints(42,48,shape2);
				convexHull(lefteye, lefteye);
				convexHull(righteye, righteye);
				std::vector<std::vector<cv::Point>> contours;
				contours.push_back(lefteye);
				contours.push_back(righteye);
				drawContours(frame,contours,-1,(255,255,255));

				double percent;
				cv::Point origin; origin.x = 20; origin.y = 40;
				int font_face = cv::FONT_HERSHEY_COMPLEX;
				char s[20];
				double t = getDistance(lefteye[1], lefteye[5]),
					y = getDistance(lefteye[2], lefteye[4]),
					u = getDistance(lefteye[0], lefteye[3]),
					g = getDistance(righteye[1], righteye[5]),
					h = getDistance(righteye[2], righteye[4]),
					j = getDistance(righteye[0], righteye[3]);

				percent = (cal_percentage(g, h, j)+cal_percentage(t,y,u))/2;

				sprintf(s, "Percentage: %.2lf", percent);
				putText(frame, s,origin, font_face, 1, cv::Scalar(0, 0, 0), 2, 8, 0);
			}

			//system("pause");
		}


		imshow("capture", frame);
		char c = (char)waitKey(10);
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
		


}