#include <mmintrin.h> /* MMX */
#include <xmmintrin.h> /* SSE,  mmintrin.h */
#include <emmintrin.h> /* SSE2, xmmintrin.h */
#include <pmmintrin.h> /* SSE3, emmintrin.h */
#include <smmintrin.h> /* SSE4.1 */
#include <nmmintrin.h> /* SSE4.2 */
#include <immintrin.h> /* AVX */
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <cmath>
#include <time.h>

using namespace cv;
using namespace std;
Mat img, image;

int pixelMatrix[3][3] = { 0 };
float GY[6] = { -1,-2,-1,1,2,1 };
float GX[6] = { 1,-1,2,-2,1,-1 };
//float GY[6] = { -1,-2,-1, 0,2, 1 };
//float GX[6] = {  0,-1, 2,-2,0,-1 };
double GY_d[6] = { -1,-2,-1,1,2,1 };
double GX_d[6] = { 1,-1,2,-2,1,-1 };

int open();
int save();
void show();
void sobel(int type);
double convolution();
double add_sse(float *m_gy, float *m_gx);
double add_avx(double *m_gy, double *m_gx);
//float add(float *a, float *b);

int main() {

	open();
	unsigned int start_time = clock(); 
	sobel(1);
	unsigned int end_time = clock(); 
	save();
	cout << "Without optimization: " << (end_time - start_time) / 1000.0 << " seconds." << endl;

	open();
	unsigned int start_time1 = clock(); 
	sobel(2);
	unsigned int end_time1 = clock(); 
	save();
	cout << "With SSE: " << (end_time1 - start_time1) / 1000.0 << " seconds." << endl;

	open();
	unsigned int start_time2 = clock(); 
	sobel(2);
	unsigned int end_time2 = clock(); 
	save();
	cout << "With AVX: " << (end_time2 - start_time2) / 1000.0 << " seconds." << endl;

	system("pause");
	//show();
}
int open() {
	img = imread("D:/1.jpg", CV_LOAD_IMAGE_COLOR); //read the image data in the file "MyPic.JPG" and store it in 'img'

	if (img.empty()) {
		cout << "Error : Image cannot be loaded..!!" << endl;
		//system("pause"); //wait for a key press
		return -1;
	}

}
int save() {
	bool bSuccess = imwrite("D:/TestImage.jpg", image); //write the image to file

	if (!bSuccess) {

		cout << "ERROR : Failed to save the image" << endl;

		//system("pause"); //wait for a key press
		return -1;
	}

}
void show() {
	namedWindow("MyWindow", CV_WINDOW_NORMAL); //create a window with the name "MyWindow"
	imshow("MyWindow", image); //display the image which is stored in the 'img' in the "MyWindow" window

	waitKey(0); //wait infinite time for a keypress

	destroyWindow("MyWindow"); //destroy the window with the name, "MyWindow"
}
void sobel(int type) {
	int edge;
	float gy[6], gx[6];
	int **a = new int *[20000];
	for (int i = 0; i != 20000; ++i) {
		a[i] = new int[20000];
	}
	image = img;
	#pragma omp parallel

	for (int i = 0; i<img.cols; i++) {
		#pragma omp for schedule(static)

		for (int j = 0; j<img.rows; j++) {
			// get pixel
			Vec3b color = image.at<Vec3b>(Point(i, j));
			int red = color.val[2];
			a[i][j] = red;
		}
	}

	#pragma omp parallel
	for (int i = 1; i < img.cols - 1; i++) {
		#pragma omp for schedule(static)
		for (int j = 1; j < img.rows - 1; j++) {
			pixelMatrix[0][0] = a[i - 1][j - 1];
			pixelMatrix[0][1] = a[i - 1][j];
			pixelMatrix[0][2] = a[i - 1][j + 1];


			pixelMatrix[1][0] = a[i][j - 1];
			pixelMatrix[1][2] = a[i][j + 1];


			pixelMatrix[2][0] = a[i + 1][j - 1];
			pixelMatrix[2][1] = a[i + 1][j];
			pixelMatrix[2][2] = a[i + 1][j + 1];

			switch (type) {
			case 1: {
				edge = (int)convolution();
			}break;
			case 2: {
				float gy[6] = { (float)pixelMatrix[0][0],(float)pixelMatrix[0][1],(float)pixelMatrix[0][2],(float)pixelMatrix[2][0],
					(float)pixelMatrix[2][1],(float)pixelMatrix[2][2] };
				float gx[6] = { (float)pixelMatrix[0][0],(float)pixelMatrix[0][2],(float)pixelMatrix[1][0], (float)pixelMatrix[1][2],
					(float)pixelMatrix[2][0],(float)pixelMatrix[2][2] };
				edge = (int)add_sse(gy, gx);
			}break;
			case 3: {
				double gy_1[6] = { (double)pixelMatrix[0][0],(double)pixelMatrix[0][1],(double)pixelMatrix[0][2],(double)pixelMatrix[2][0],
					(double)pixelMatrix[2][1],(double)pixelMatrix[2][2] };
				double gx_1[6] = { (double)pixelMatrix[0][0],(double)pixelMatrix[0][2],(double)pixelMatrix[1][0], (double)pixelMatrix[1][2],
					(double)pixelMatrix[2][0],(double)pixelMatrix[2][2] };
				edge = (int)add_avx(gy_1, gx_1);
			}break;
			}
			Vec3b color = image.at<Vec3b>(Point(i, j));
			color.val[0] = edge;
			color.val[1] = edge;
			color.val[2] = edge;

			// set pixel 
			image.at<Vec3b>(Point(i, j)) = color;
		}
	}
}
double convolution() {

	int gy = (pixelMatrix[0][0] = pixelMatrix[0][0] * -1) + (pixelMatrix[0][1] = pixelMatrix[0][1] * -2) + (pixelMatrix[0][2] = pixelMatrix[0][2] * -1) + (pixelMatrix[2][0]) + (pixelMatrix[2][1] = pixelMatrix[2][1] * 2) + (pixelMatrix[2][2] = pixelMatrix[2][2] * 1);
	int gx = (pixelMatrix[0][0]) + (pixelMatrix[0][2] = pixelMatrix[0][2] * -1) + (pixelMatrix[1][0] = pixelMatrix[1][0] * 2) + (pixelMatrix[1][2] = pixelMatrix[1][2] * -2) + (pixelMatrix[2][0]) + (pixelMatrix[2][2] = pixelMatrix[2][2] * -1);

	return sqrt(pow(gy, 2) + pow(gx, 2));
}
double add_sse(float *m_gy, float *m_gx) { //SSE

	__m128 y0, y1, x1, x0, gy0, gy1, gx1, gx0;
	float Y0[4], Y1[4], X0[4], X1[4];
	
	y0 = _mm_set_ps(m_gy[0], m_gy[1], m_gy[2], m_gy[3]);
	y1 = _mm_set_ps(m_gy[4], m_gy[5], 0.0, 0.0);

	x0 = _mm_set_ps(m_gx[0], m_gx[1], m_gx[2], m_gx[3]);
	x1 = _mm_set_ps(m_gx[4], m_gx[5], 0.0, 0.0);

	gy0 = _mm_set_ps(GY[0], GY[1], GY[2], GY[3]);
	gy1 = _mm_set_ps(GY[4], GY[5], 0.0, 0.0);

	gx0 = _mm_set_ps(GX[0], GX[1], GX[2], GX[3]);
	gx1 = _mm_set_ps(GX[4], GX[5], 0.0, 0.0);

	y0 = _mm_mul_ps(y0, gy0);
	y1 = _mm_mul_ps(y1, gy1);

	x0 = _mm_mul_ps(x0, gx0);
	x1 = _mm_mul_ps(x1, gx1);

	_mm_store_ps(Y0, y0);
	_mm_store_ps(Y1, y1);

	_mm_store_ps(X0, x0);
	_mm_store_ps(X1, x1);

	double sum_gy = 0, sum_gx = 0;
	
	for (int i = 0; i < 4; i++) {
		sum_gx = sum_gx + X0[i];
		sum_gx = sum_gx + X1[i];
		sum_gy = sum_gy + Y0[i];
		sum_gy = sum_gy + Y1[i];

	}
	return sqrt(pow(sum_gy, 2) + pow(sum_gx, 2));
}


double add_avx(double *m_gy, double *m_gx) { //AVX

	__m256d y0, y1, x1, x0, gy0, gy1, gx1, gx0;
	double Y0[4], Y1[4], X0[4], X1[4];

	y0 = _mm256_setr_pd(m_gy[0], m_gy[1], m_gy[2], m_gy[3]);
	y1 = _mm256_setr_pd(m_gy[4], m_gy[5], 0.0, 0.0);

	x0 = _mm256_setr_pd(m_gx[0], m_gx[1], m_gx[2], m_gx[3]);
	x1 = _mm256_setr_pd(m_gx[4], m_gx[5], 0.0, 0.0);

	gy0 = _mm256_setr_pd(GY[0], GY[1], GY[2], GY[3]);
	gy1 = _mm256_setr_pd(GY[4], GY[5], 0.0, 0.0);

	gx0 = _mm256_setr_pd(GX[0], GX[1], GX[2], GX[3]);
	gx1 = _mm256_setr_pd(GX[4], GX[5], 0.0, 0.0);

	y0 = _mm256_mul_pd(y0, gy0);
	y1 = _mm256_mul_pd(y1, gy1);

	x0 = _mm256_mul_pd(x0, gx0);
	x1 = _mm256_mul_pd(x1, gx1);

	_mm256_stream_pd(Y0, y0);
	_mm256_stream_pd(Y1, y1);

	_mm256_stream_pd(X0, x0);
	_mm256_stream_pd(X1, x1);

	double sum_gy = 0, sum_gx = 0;
	
	for (int i = 0; i < 4; i++) {
		sum_gx = sum_gx + X0[i];
		sum_gx = sum_gx + X1[i];
		sum_gy = sum_gy + Y0[i];
		sum_gy = sum_gy + Y1[i];

	}
	return sqrt(pow(sum_gy, 2) + pow(sum_gx, 2));
}