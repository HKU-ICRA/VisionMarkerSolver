#include <vector>
#include <string>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "detect.h"


using namespace std;
using namespace cv;


int main (int argc, char** argv) {
	Mat image = imread("C:\\Users\\impec\\Pictures\\markertest.jpg", IMREAD_COLOR);
	vector<string> vss;
	vector<vector<Point> > vps;
	detectMarker(image, vss, vps);
	for (int i(0); i < vss.size(); ++i) {
		cout << vss[i] << "\n";
	}
	return 0;
}
