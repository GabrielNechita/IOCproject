// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "Functions.h"
#include <queue>
#include <opencv2/video/tracking.hpp>
#include <random>
using namespace std;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // no dword alignment is done !!!
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
				/* sau puteti scrie:
				uchar val = lpSrc[i*width + j];
				lpDst[i*width + j] = 255 - val;
				//	w = width pt. imagini cu 8 biti / pixel
				//	w = 3*width pt. imagini cu 24 biti / pixel
				*/
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);
		
		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // latimea in octeti a unei linii de imagine
		
		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);
		
		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* dstDataPtrH = dstH.data;
		uchar* dstDataPtrS = dstS.data;
		uchar* dstDataPtrV = dstV.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				// sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
				int gi = i*width + j;
				
				dstDataPtrH[gi] = hsvDataPtr[hi] * 510/360;		// H = 0 .. 255
				dstDataPtrS[gi] = hsvDataPtr[hi + 1];			// S = 0 .. 255
				dstDataPtrV[gi] = hsvDataPtr[hi + 2];			// V = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);
		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		double pL = k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}
//lab 2
void RGB2HSV(){
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		Mat hsv;

		Mat channels[3];

		cvtColor(src, hsv, CV_BGR2HSV);
		split(hsv, channels);

		Mat H(src.rows, src.cols, CV_8UC1);
		Mat S(src.rows, src.cols, CV_8UC1);
		Mat V(src.rows, src.cols, CV_8UC1);

		H = channels[0] * 510 / 360;
		S = channels[1];
		V = channels[2];

		imshow("src", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		int HistH[256] = { 0 };
		int HistS[256] = { 0 };
		int HistV[256] = { 0 };

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				uchar value = H.at<uchar>(i, j);
				HistH[value]++;

				value = S.at<uchar>(i, j);
				HistS[value]++;

				value = V.at<uchar>(i, j);
				HistV[value]++;
			}
		}

		showHistogram("HistH", HistH, 256, 256, true);
		showHistogram("HistS", HistS, 256, 256, true);
		showHistogram("HistV", HistV, 256, 256, true);

		Mat binarizareH(H.rows, H.cols, CV_8UC1);
		for (int i = 0; i < H.rows; i++)
		{
			for (int j = 0; j < H.cols; j++)
			{
				if (H.at<uchar>(i, j) > 30)
					binarizareH.at<uchar>(i, j) = 0;
				else
					binarizareH.at<uchar>(i, j) = 255;
			}
		}
		imshow("binarizareH", binarizareH);
		waitKey(0);
	}
}


void lab5(){
	char fname[MAX_PATH];
	while (openFileDlg(fname)){
		Mat src_image = imread(fname, CV_LOAD_IMAGE_COLOR);
		Mat dst_image = src_image.clone();
		cvtColor(src_image, src_image, CV_BGR2GRAY);
		GaussianBlur(src_image, src_image, Size(5, 5), 0, 0);

		vector<Point2f> corners;
		int maxCorners = 100;
		double qualityLevel = 0.01;
		double minDistance = 10;
		int blockSize = 3; // 2,3, ...
		bool useHarrisDetector = true;
		double k = 0.04;
		goodFeaturesToTrack(src_image,
			corners,
			maxCorners,
			qualityLevel,
			minDistance,
			Mat(),
			blockSize,
			useHarrisDetector,
			k);
		for (Point2f p : corners)
		{
			circle(dst_image, p, 3, Scalar(0, 255, 0));
		}

		imshow("dst", dst_image);
		waitKey(0);

		Size winSize = Size(5, 5);
		Size zeroZone = Size(-1, -1);

		TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
		cornerSubPix(src_image, corners, winSize, zeroZone, criteria);

		FILE *f = fopen("out.txt", "w");
		for (int i = 0; i < corners.size(); i++)
		{
			fprintf(f, "(%f, %f)\n", corners[i].x, corners[i].y);
		}
		fclose(f);

	}
}

Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;

char* source_window = "Source image";
char* corners_window = "Corners detected";

void cornerHarris_demo(int, void*)
{

	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);

	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	/// Detecting corners
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	/// Drawing a circle around corners
	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			if ((int)dst_norm.at<float>(j, i) > thresh)
			{
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
			}
		}
	}
	/// Showing the result
	namedWindow(corners_window, CV_WINDOW_AUTOSIZE);
	imshow(corners_window, dst_norm_scaled);
}

void lab5_4(){

	char fname[MAX_PATH];
	while (openFileDlg(fname)){
		src = imread(fname, 1);
		cvtColor(src, src_gray, CV_BGR2GRAY);

		/// Create a window and a trackbar
		namedWindow(source_window, CV_WINDOW_AUTOSIZE);
		createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo);
		imshow(source_window, src);

		cornerHarris_demo(0, 0);

		waitKey(0);
	}
}

void lab5_5(){
	VideoCapture cap("Images/Videos/BS.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);

		Mat dst_image = frame.clone();
		cvtColor(frame, frame, CV_BGR2GRAY);
		GaussianBlur(frame, frame, Size(5, 5), 0, 0);

		vector<Point2f> corners;
		int maxCorners = 100;
		double qualityLevel = 0.01;
		double minDistance = 10;
		int blockSize = 3; // 2,3, ...
		bool useHarrisDetector = true;
		double k = 0.04;
		goodFeaturesToTrack(frame,
			corners,
			maxCorners,
			qualityLevel,
			minDistance,
			Mat(),
			blockSize,
			useHarrisDetector,
			k);
		for (Point2f p : corners)
		{
			circle(dst_image, p, 3, Scalar(0, 255, 0));
		}

		imshow("dst", dst_image);
		waitKey(0);

		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}

void lab6(){
	VideoCapture cap("Images/Videos/laboratory.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat frame, gray; //current frame: original and gray
	Mat backgnd; // background model
	Mat diff; //difference image: |frame_gray - bacgnd|
	Mat dst; //output image/frame
	char c;
	int frameNum = -1; //current frame counter
	const int method = 3;
	// method =
	// 1 - frame difference
	// 2 - running average
	// 3 - running average with selectivity
	const unsigned char Th = 25;
	const double alpha = 0.05;


	for (;;){
		cap >> frame; // achizitie frame nou
		if (frame.empty())
		{
			printf("End of video file\n");
			break;
		}
		++frameNum;
		if (frameNum == 0)
			imshow("sursa", frame); // daca este primul cadru se afiseaza doar sursa
		cvtColor(frame, gray, CV_BGR2GRAY);
		//Optional puteti aplica si un FTJ Gaussian
		//Se initializeaza matricea / imaginea destinatie pentru fiecare frame
		//dst=gray.clone();
		// sau
		//
		dst = Mat::zeros(gray.size(), gray.type());
		const int channels_gray = gray.channels();

		double t = (double)getTickCount(); // Get the current time [s]

		//restrictionam utilizarea metodei doar pt. imagini grayscale cu un canal (8 bit / pixel)
		if (channels_gray > 1)
			return;
		if (frameNum > 0) // daca nu este primul cadru
		{
			absdiff(gray, backgnd, diff);
			//------ SABLON DE PRELUCRARI PT. METODELE BACKGROUND SUBTRACTION -------
			// Calcul imagine diferenta dintre cadrul current (gray) si fundal (backgnd)
			// Rezultatul se pune in matricea/imaginea diff
			// Se actualizeaza matricea/imaginea model a fundalului (backgnd)
			// conform celor 3 metode:
			if (method == 1)
				backgnd = gray.clone();
			if (method == 2)
				addWeighted(gray, alpha, backgnd, 1.0 - alpha, 0, backgnd);


			// met 1: backgnd = gray.clone();
			// met 2: addWeighted(gray, alpha, backgnd, 1.0-alpha, 0, backgnd);
			// Pt. met 3 (running average with selectivity) se recomanda actualizarea
			// pixelilor de fundal in mod individual in bucla de parcurgere de mai jos 5

			for (int i = 0; i < diff.rows; i++)
			{
				for (int j = 0; j < diff.cols; j++)
				{
					if (diff.at<uchar>(i, j) > Th)
					{
						dst.at<uchar>(i, j) = 255;
					}
					else{
						if (method == 3)
						{
							backgnd.at<uchar>(i, j) =
								alpha*gray.at<uchar>(i, j) + (1.0 - alpha)*backgnd.at<uchar>(i, j);
							dst.at<uchar>(i, j) = 0;
						}
					}
				}
			}
			Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
			erode(dst, dst, element, Point(-1, -1), 2);
			dilate(dst, dst, element, Point(-1, -1), 2);


			// Se parcurge sistematic matricea diff
			//daca valoarea pt. pixelul current diff.at<uchar>(i,j) > Th
			// marcheaza pixelul din imaginea destinatie ca obiect:
			//dst.at<uchar>(i, j) = 255 // pentru toate metodele
			// altfel
			// actualizeaza model background (doar pt. metoda 3)
			//-------------------------------------------------------------------------
			// Afiseaza imaginea sursa si destinatie
			imshow("sursa", frame); // show source
			imshow("dest", dst); // show destination
			imshow("diff", diff);
			// Plasati aici codul pt. vizualizarea oricaror rezultate intermediare
			// Ex: afisarea intr-o fereastra noua a imaginii diff
		}
		else // daca este primul cadru, modelul de fundal este chiar el
			backgnd = gray.clone();

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms]
		printf("%d - %.3f [ms]\n", frameNum, t * 1000);

		// Conditia de avansare/terminare in cilului for(;;) de procesare
		c = cvWaitKey(0); // press any key to advance between frames
		//for continous play use cvWaitKey( delay > 0)
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - playback finished\n");
			break; //ESC pressed
		}
	}


}

void lab7(){
	VideoCapture cap("Images/Videos/S3.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
	char c;
	Mat frame, crnt; // crntent frames: original (frame) and converted to grayscale (crnt)
	Mat prev; // previous frame (grayscale)
	Mat dst; // output image/frame
	Mat flow; // flow - matrix containing the optical flow vectors/pixel
	Mat flow1;


	int frameNum = -1; //crntent frame counter
	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of video file\n");
			break;
		}
		++frameNum;
		//. . .
		// functii de pre-procesare
		cvtColor(frame, crnt, CV_BGR2GRAY);
		GaussianBlur(crnt, crnt, Size(5, 5), 0.8, 0.8);
		//. . .
		if (frameNum > 0) // not the first frame
		{
			Mat flow1; Mat flow2;
			//. . . .
			// functii de procesare (calcul flux optic) si afisare
			//. . .

			// Horn-Shunk
			calcOpticalFlowHS(prev, crnt, 0, 0.1, TermCriteria(TermCriteria::MAX_ITER, 16, 0), flow1);
			// Lukas-Kanade
			calcOpticalFlowLK(prev, crnt, Size(15, 15), flow2);
			showFlow("HS", prev, flow1, 1, 4, true, true, false);
			showFlow("LK", prev, flow2, 1, 4, true, true, false);


			// parameters for calcOpticalFlowPyrLK
			vector<Point2f> prev_pts; // vector of 2D points with previous image features
			vector<Point2f> crnt_pts;// vector of 2D points with current image (matched) features
			vector<uchar> status; // output status vector: 1 if the wlow for the corresponding
			//feature was found. 0 otherwise
			vector<float> error; // output vector of errors; each element of the vector is set to
			//an error for the corresponding feature
			Size winSize = Size(21, 21); // size of the search window at each pyramid level - deafult
			//(21, 21)
			int maxLevel = 3; // maximal pyramid level number - deafult 3
			//parameter, specifying the termination criteria of the iterative search algorithm
			// (after the specified maximum number of iterations criteria.maxCount or when the search
			//window moves by less than criteria.epsilon
			// deafult 30, 0.01
			TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 20, 0.03);
			int flags = 0;
			double minEigThreshold = 1e-4;

			// Apply corner detection

			int maxCorners = 100;
			double qualityLevel = 0.01;
			double minDistance = 10;
			int blockSize = 3; // 2,3, ...
			bool useHarrisDetector = true;
			double k = 0.04;
			goodFeaturesToTrack(prev,
				prev_pts,
				maxCorners,
				qualityLevel,
				minDistance,
				Mat(),
				blockSize,
				useHarrisDetector,
				k);

			calcOpticalFlowPyrLK(prev, crnt, prev_pts, crnt_pts, status, error, winSize, maxLevel, criteria);
			showFlowSparse("Dst", prev, prev_pts, crnt_pts, status, error, 2, true, true, true);
			//prev_pts = crnt_pts;
		}
		// store crntent frame as previos for the next cycle
		prev = crnt.clone();


		c = cvWaitKey(0); // press any key to advance between frames
		//for continous play use cvWaitKey( delay > 0)
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - playback finished\n\n");
			break; //ESC pressed
		}

	}

}

void L3(){
	float k;
	printf("k = ");
	scanf("%f", &k);

	const int u = 16;
	const int t = 5;

	Mat src;
	Mat hsv;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		GaussianBlur(src, src, Size(5, 5), 0, 0);

		Mat hsvImg = src.clone();
		cvtColor(src, hsvImg, CV_BGR2HSV);

		Mat channels[3];
		split(hsvImg, channels);

		Mat dst = Mat::zeros(channels[0].size(), CV_8UC1);

		Mat hue = Mat::zeros(channels[0].size(), CV_8UC1);
		hue = channels[0] * (255 / 180);

		for (int i = 0; i < hue.rows; i++){
			for (int j = 0; j < hue.cols; j++){

				uchar pixelH = hue.at<uchar>(i, j);
				if (pixelH > u - k * t && pixelH < u + k * t){
					dst.at<uchar>(i, j) = 225;
				}
				else
					dst.at<uchar>(i, j) = 0;
			}
		}


		imshow("dstA", dst);

		Mat element2 = getStructuringElement(MORPH_RECT, Size(5, 5));
		erode(dst, dst, element2, Point(-1, -1), 2);
		dilate(dst, dst, element2, Point(-1, -1), 4);
		erode(dst, dst, element2, Point(-1, -1), 2);

		imshow("dstB", dst);

		Labeling("contur fara  linie", dst, false, false);

	
		Labeling("contur cu linie", dst, false, true);

		waitKey(0);
	}

}

bool isInside(Mat src, int x, int y){
	if ((x >= 0 && x < src.cols) && (y >= 0 && y < src.rows)){
		return true;
	}
	return false;
}

void CallBackL4(int event, int x, int y, int flags, void *userdata){

	Mat *H = (Mat*)userdata;
	if (event == EVENT_LBUTTONDOWN){

		int width = (*H).cols;
		int height = (*H).rows;

		Mat labels = Mat::zeros((*H).size(), CV_16UC1);
		Mat dst = Mat::zeros((*H).size(), CV_8UC1);
		queue<Point> que;
		double avg = (*H).at<uchar>(y, x);

		int k = 1;
		int N = 1;
		float T = 15;

		labels.at<ushort>(y, x) = k;

		que.push(Point(x, y));
		int nghbrsX[] = { -1, -1, -1, 0, 1, 1, 1, 0 };
		int nghbrsY[] = { -1, 0, 1, 1, 1, 0, -1, -1 };

		while (!que.empty()){
			Point oldest = que.front();
			que.pop();

			int xx = oldest.x;
			int yy = oldest.y;

			for (int i = 0; i < 8; i++){

				if (isInside(*H, xx + nghbrsX[i], yy + nghbrsY[i])){
					if ((abs((*H).at<uchar>(yy + nghbrsY[i], xx + nghbrsX[i]) - avg) < T) &&
						(labels.at<ushort>(yy + nghbrsY[i], xx + nghbrsX[i]) == 0)){

						que.push(Point(xx + nghbrsX[i], yy + nghbrsY[i]));
						labels.at<ushort>(yy + nghbrsY[i], xx + nghbrsX[i]) = k;
						avg = (N*avg + (*H).at<uchar>(yy + nghbrsY[i], xx + nghbrsX[i])) / (N + 1);
						N++;
					}
				}
			}
		}

		for (int i = 0; i < height; i++){
			for (int j = 0; j < width; j++)
				if (labels.at<ushort>(i, j) == 1)
					dst.at<uchar>(i, j) = 255;
		}

		Mat element1 = getStructuringElement(MORPH_CROSS, Size(5, 5));
		erode(dst, dst, element1, Point(-1, -1), 2);
		dilate(dst, dst, element1, Point(-1, -1), 4);
		erode(dst, dst, element1, Point(-1, -1), 2);

		imshow("dst", dst);
		waitKey();
	}

}


void PrCom(const string& name, void *userdata, void *userdata2, float T){

	Mat *H = (Mat*)userdata;
	Mat *S = (Mat*)userdata2;

	int width = (*H).cols;
	int height = (*H).rows;

	Mat labels = Mat::zeros((*H).size(), CV_16UC1);
	queue<Point> que;
	int k = 0;
	int N = 1;

	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++)
		{
			if (labels.at<ushort>(y, x) == 0)
			{
				k++;
				double avgH = (*H).at<uchar>(y, x);
				double avgS = (*S).at<uchar>(y, x);

				labels.at<ushort>(y, x) = k;

				que.push(Point(x, y));
				int nghbrsX[] = { -1, -1, -1, 0, 1, 1, 1, 0 };
				int nghbrsY[] = { -1, 0, 1, 1, 1, 0, -1, -1 };

				while (!que.empty()){
					Point oldest = que.front();
					que.pop();

					int xx = oldest.x;
					int yy = oldest.y;

					for (int i = 0; i < 8; i++){

						if (isInside(*H, xx + nghbrsX[i], yy + nghbrsY[i])){

							if ((sqrt( 
								pow(((*H).at<uchar>(yy + nghbrsY[i], xx + nghbrsX[i]) - avgH), 2) + 
								pow(((*S).at<uchar>(yy + nghbrsY[i], xx + nghbrsX[i]) - avgS), 2)
									
								) < T) && (labels.at<ushort>(yy + nghbrsY[i], xx + nghbrsX[i]) == 0)){

								que.push(Point(xx + nghbrsX[i], yy + nghbrsY[i]));
								labels.at<ushort>(yy + nghbrsY[i], xx + nghbrsX[i]) = k;
								avgH = (N*avgH + (*H).at<uchar>(yy + nghbrsY[i], xx + nghbrsX[i])) / (N + 1);
								avgS = (N*avgS + (*S).at<uchar>(yy + nghbrsY[i], xx + nghbrsX[i])) / (N + 1);
								N++;
							}
						}
					}
				}
				//printf("\navg:%f N:%d", avg, N);
			}

		}
	}

	Mat colored((*H).size(), CV_8UC3);
	std::default_random_engine gen1;
	std::uniform_int_distribution<int> d(0, 255);
	Vec3b colors[10000];
	for (int i = 1; i <= k; i++)
	{
		colors[i][0] = d(gen1);
		colors[i][1] = d(gen1);
		colors[i][2] = d(gen1);
	}
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (labels.at<ushort>(i, j) > 0) {
				int x = labels.at<ushort>(i, j);
				colored.at<Vec3b>(i, j) = colors[x];
			}
		}
	}

	imshow(name, colored);
}

void PrL4(const string& name, void *userdata, float T){

	Mat *H = (Mat*)userdata;
	

		int width = (*H).cols;
		int height = (*H).rows;

		Mat labels = Mat::zeros((*H).size(), CV_16UC1);
		Mat dst = Mat::zeros((*H).size(), CV_8UC1);
		queue<Point> que;
		int k = 0;
		int N = 1;

		for (int y = 0; y < height; y++){
			for (int x = 0; x < width; x++)
			{
				if (labels.at<ushort>(y, x) == 0)
				{
					k++;
					double avg = (*H).at<uchar>(y, x);

					labels.at<ushort>(y, x) = k;

					que.push(Point(x, y));
					int nghbrsX[] = { -1, -1, -1, 0, 1, 1, 1, 0 };
					int nghbrsY[] = { -1, 0, 1, 1, 1, 0, -1, -1 };

					while (!que.empty()){
						Point oldest = que.front();
						que.pop();

						int xx = oldest.x;
						int yy = oldest.y;

						for (int i = 0; i < 8; i++){

							if (isInside(*H, xx + nghbrsX[i], yy + nghbrsY[i])){
								if ((abs((*H).at<uchar>(yy + nghbrsY[i], xx + nghbrsX[i]) - avg) < T) &&
									(labels.at<ushort>(yy + nghbrsY[i], xx + nghbrsX[i]) == 0)){

									que.push(Point(xx + nghbrsX[i], yy + nghbrsY[i]));
									labels.at<ushort>(yy + nghbrsY[i], xx + nghbrsX[i]) = k;
									avg = (N*avg + (*H).at<uchar>(yy + nghbrsY[i], xx + nghbrsX[i])) / (N + 1);
									N++;
								}
							}
						}
					}
					//printf("\navg:%f N:%d", avg, N);
				}
				
			}
		}

		Mat colored((*H).size(), CV_8UC3);
		std::default_random_engine gen1;
		std::uniform_int_distribution<int> d(0, 255);
		Vec3b colors[9000];
		for (int i = 1; i <= k; i++)
		{
			colors[i][0] = d(gen1);
			colors[i][1] = d(gen1);
			colors[i][2] = d(gen1);
		}
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (labels.at<ushort>(i, j) > 0) {
					int x = labels.at<ushort>(i, j);
					colored.at<Vec3b>(i, j) = colors[x];
				}
			}
		}

		imshow(name, colored);
}

void pr(){
	Mat src;
	Mat hsv;

	char fname[MAX_PATH];
	while (openFileDlg(fname)){

		src = imread(fname);
		imshow("src", src);
		//GaussianBlur(src, src, Size(5, 5), 0, 0);
		cvtColor(src, hsv, CV_BGR2HSV);
		Mat channels[3];
		split(hsv, channels);
		Mat dstH = Mat(src.rows, src.cols, CV_8UC1);
		Mat dstS = Mat(src.rows, src.cols, CV_8UC1);
		dstH = channels[0] * 255 / 180;
		dstS = channels[1];

		//imshow("H", dstH);
		//imshow("S", dstS);
		
		double t1 = (double)getTickCount(); // Get the crntent time [s]
		PrL4("H", &dstH, 5);
		// Get the crntent time again and compute the time difference [s]
		t1 = ((double)getTickCount() - t1) / getTickFrequency();
		// Print (in the console window) the processing time in [ms]
		printf("H: %.3f [ms]\n", t1 * 1000);

		double t2 = (double)getTickCount(); // Get the crntent time [s]
		PrL4("S", &dstS, 5);
		// Get the crntent time again and compute the time difference [s]
		t2 = ((double)getTickCount() - t2) / getTickFrequency();
		// Print (in the console window) the processing time in [ms]
		printf("S: %.3f [ms]\n", t2 * 1000);
		
		double t3 = (double)getTickCount(); // Get the crntent time [s]
		PrCom("H&S", &dstH, &dstS, 30);
		// Get the crntent time again and compute the time difference [s]
		t3 = ((double)getTickCount() - t3) / getTickFrequency();
		// Print (in the console window) the processing time in [ms]
		printf("H&S: %.3f [ms]\n", t3 * 1000);

		waitKey(0);
	}
}

void etichetarePI() {
	char fname[10000];

	while (openFileDlg(fname))
	{
		int label = 0;
		Mat img = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		imshow("original image", img);
		Mat labels(img.rows, img.cols, CV_32SC1);
		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				labels.at<int>(i, j) = 0;
			}
		}
		Mat colored(img.rows, img.cols, CV_8UC3);
		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				if (img.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
					label++;
					std::queue<Point2i> Q;
					labels.at<int>(i, j) = label;
					Q.push({ i, j });
					while (!Q.empty()) {
						Point2i p = Q.front();
						Q.pop();
						int dx[10] = { -1, 0, 1, -1, 1, -1, 0, 1 };
						int dy[10] = { -1, -1, -1, 0, 0, 1, 1, 1 };

						for (int k = 0; k < 8; k++) {
							if (img.at<uchar>(p.x + dx[k], p.y + dy[k]) == 0 && labels.at<int>(p.x + dx[k], p.y + dy[k]) == 0) {
								labels.at<int>(p.x + dx[k], p.y + dy[k]) = label;
								Q.push({ p.x + dx[k], p.y + dy[k] });
							}
						}


					}

				}
			}
		}
		default_random_engine gen;
		std::uniform_int_distribution<int> d(0, 255);
		Vec3b colors[1000];
		for (int i = 1; i <= label; i++)
		{
			colors[i][0] = d(gen);
			colors[i][1] = d(gen);
			colors[i][2] = d(gen);
		}
		for (int i = 0; i < img.rows; i++)
		{
			for (int j = 0; j < img.cols; j++)
			{
				if (labels.at<int>(i, j) > 0) {
					colored.at<Vec3b>(i, j) = colors[labels.at<int>(i, j)];
				}
			}
		}
		imshow("colored labeled img", colored);
		waitKey(0);
	}
}

void L4_RG(){


	Mat src;
	Mat hsv;

	char fname[MAX_PATH];
	while (openFileDlg(fname)){

		src = imread(fname);

		GaussianBlur(src, src, Size(5, 5), 0, 0);
		cvtColor(src, hsv, CV_BGR2HSV);
		Mat channel[3];
		split(hsv, channel);
		Mat dstH = Mat(src.rows, src.cols, CV_8UC1);
		dstH = channel[0] * 255 / 180;

		namedWindow("src", 1);
		setMouseCallback("src", CallBackL4, &dstH);
		imshow("src", src);
		waitKey(0);
	}
}

void l()
{
	Mat src;
	Mat hsv;

	char fname[MAX_PATH];
	while (openFileDlg(fname)){
		src = imread(fname, CV_8UC1);
		Labeling("labeling", src, false, false);
		waitKey(0);
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - RGB2HSV\n");
		printf(" 11 - Lab3\n");

		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				RGB2HSV();
				break;
			case 11:
				L3();
				break;
			case 12:
				L4_RG();
				break;
			case 13:
				lab5();
				break;
			case 14:
				lab5_4();
				break;
			case 15:
				lab5_5();
				break;
			case 16:
				lab6();
				break;
			case 17:
				lab7();
				break;
			case 18:
				pr();
				break;
			case 19:
				etichetarePI();
				break;
				

		}
	}
	while (op!=0);
	return 0;
}