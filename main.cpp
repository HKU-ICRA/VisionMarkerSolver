#include <vector>
#include <string>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "detect.h"


using namespace std;
using namespace cv;


int main (int argc, char** argv) {
	VideoCapture cap(0);
	while (!cap.isOpened()) {
		VideoCapture cap(0);
	}
	while (true) {
		Mat frame;
		Mat dframe;
		cap >> frame;
		vector<string> vss;
		vector<vector<Point> > vps;
		detectMarker(frame, dframe, vss, vps);
		for (int i(0); i < vss.size(); ++i) {
			cout << vss[i] << "\n";
		}
		imshow("Frame", frame);
		if (!dframe.empty())
			imshow("Dframe", dframe);
		if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
	}
	return 0;
}
