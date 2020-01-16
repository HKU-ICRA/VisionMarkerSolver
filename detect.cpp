#include <cmath>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "detect.h"
#include "markerDict.h"


using namespace std;
using namespace cv;


/* Tuneable variables

	mcolor_lb = lowerbound threshold for marker detection
	mcolor_ub = upperbound threshold for marker detection
	COVERAGE = percentage of white pixels within cell to be classified as "filled"
	IGNORE_MARGIN = The percentage of cell size to be trimmed away
	MIN_WARP_WH = Minimum width and height of warped image to not be rejected as candidate
*/
Scalar mcolor_lb(10, 100, 100);
Scalar mcolor_ub(30, 255, 255);
double COVERAGE = 0.5;
double IGNORE_MARGIN = 0.2;
int MIN_WARP_WH = 21;	// this should be >= 7
int MAX_ORDEREDRECTS = 25;


/* Constructors */
vector<vector<Point> > orderRects(vector<vector<Point> >& rects);
vector<Mat> perspectiveTrans(Mat img, vector<vector<Point> >& orderedRects);
vector<Mat> tile(Mat warp);


/* dictionary of markers */
markerDict dictionary;


/* main detection function */
void detectMarker(Mat img, Mat& dimg, vector<string>& marker_names, vector<vector<Point> >& marker_rects) {
	/* Threshold image */
	cvtColor(img, img, COLOR_BGR2HSV);
	inRange(img, mcolor_lb, mcolor_ub, img);
	/* Get contours */
	vector<vector<Point> > contours;
	findContours(img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
	if (contours.empty()) {
		return;
	}
	sort(contours.begin(), contours.end(), [](const vector<Point> & c1, const vector<Point> & c2) {
		return contourArea(c1, false) > contourArea(c2, false);	// Greater first
	});
	/* Get quadrilaterals */
	double epsilon = 0.01 * arcLength(contours[0], true);
	vector<vector<Point> > rects;
	for (vector<Point> vp : contours) {
		vector<Point> rect_cand;
		approxPolyDP(vp, rect_cand, epsilon, true);
		if (rect_cand.size() == 4)
			rects.push_back(rect_cand);
	}
	if (rects.empty()) {
		return;
	}
	/* Prune rectangles by removing those within existing rectangles */
	// TO-DO
	/* Perspective transform */
	vector<vector<Point> > orderedRects = orderRects(rects);
	if (orderedRects.size() > MAX_ORDEREDRECTS)
		orderedRects.resize(MAX_ORDEREDRECTS);
	vector<Mat> warps = perspectiveTrans(img, orderedRects);
	/* Otsu threshold + gaussian blur */
	for (int i(0); i < warps.size(); ++i) {
		GaussianBlur(warps[i], warps[i], Size(3, 3), 0);
		threshold(warps[i], warps[i], 0, 255, THRESH_BINARY + THRESH_OTSU);
	}
	dimg = warps[0].clone();
	/* Split image into 7x7 tiles */
	vector<vector<Mat> > grids;
	int deleted_boxes = 0;
	for (int i(0); i < warps.size(); ++i) {
		if (warps[i].rows >= MIN_WARP_WH && warps[i].cols >= MIN_WARP_WH) {
			grids.push_back(tile(warps[i]));
		}
		else {
			orderedRects.erase(orderedRects.begin() + i - deleted_boxes);
			++deleted_boxes;
		}
;	}
	if (grids.empty()) {
		return;
	}
	/* Determine whether cell is filled and return detected markers */
	for (int i(0); i < grids.size(); ++i) {
		Mat marker(7, 7, CV_32SC1);
		for (int y(0); y < 7; ++y) {
			for (int x(0); x < 7; ++x) {
				marker.at<int>(y, x) = ((double)countNonZero(grids[i][x + y * 7]) / 49.0) >= COVERAGE;
			}
		}
		string marker_name = dictionary.getMarker(marker);
		if (marker_name != "none") {
			marker_names.push_back(marker_name);
			marker_rects.push_back(orderedRects[i]);
		}
	}
}


vector<vector<Point> > orderRects(vector<vector<Point> >& rects) {
	vector<pair<double, double> > pts_sum(rects.size(), pair<double, double>(10000.0, -10000.0));	// min, max
	vector<pair<int, int> > pts_sum_idx(rects.size(), pair<int, int>(0, 0));
	vector<pair<double, double> > pts_diff(rects.size(), pair<double, double>(10000.0, -10000.0));	// min, max
	vector<pair<int, int> > pts_diff_idx(rects.size(), pair<int, int>(0, 0));
	for (int i(0); i < rects.size(); ++i) {
		for (int j(0); j < 4; ++j) {
			// Handle sums
			double sum = rects[i][j].y + rects[i][j].x;
			if (sum <= pts_sum[i].first) {
				pts_sum[i].first = sum;
				pts_sum_idx[i].first = j;
			}
			if (sum > pts_sum[i].second) {
				pts_sum[i].second = sum;
				pts_sum_idx[i].second = j;
			}
			// Handle differences
			double diff = rects[i][j].y - rects[i][j].x;
			if (diff <= pts_diff[i].first) {
				pts_diff[i].first = diff;
				pts_diff_idx[i].first = j;
			}
			if (diff > pts_diff[i].second) {
				pts_diff[i].second = diff;
				pts_diff_idx[i].second = j;
			}
		}
	}
	vector<vector<Point> > orderedRects(rects.size(), vector<Point>(4));
	for (int i(0); i < rects.size(); ++i) {
		orderedRects[i][0] = rects[i][pts_sum_idx[i].first];
		orderedRects[i][2] = rects[i][pts_sum_idx[i].second];
		orderedRects[i][1] = rects[i][pts_diff_idx[i].first];
		orderedRects[i][3] = rects[i][pts_diff_idx[i].second];
	}
	return orderedRects;
}


vector<Mat> perspectiveTrans(Mat img, vector<vector<Point> >& orderedRects) {
	vector<Mat> warps(orderedRects.size());
	for (int i(0); i < orderedRects.size(); ++i) {
		Point tl = orderedRects[i][0];
		Point tr = orderedRects[i][1];
		Point br = orderedRects[i][2];
		Point bl = orderedRects[i][3];
		double w1 = sqrtf(powf((br.y - bl.y), 2) + powf((br.x - bl.x), 2));
		double w2 = sqrtf(powf((tr.y - tl.y), 2) + powf((tr.x - tl.x), 2));
		double w = fmax(w1, w2);
		double h1 = sqrtf(powf((tr.y - br.y), 2) + powf((tr.x - br.x), 2));
		double h2 = sqrtf(powf((tl.y - bl.y), 2) + powf((tl.x - bl.x), 2));
		double h = fmax(h1, h2);
		Point2f src[4] = {
			orderedRects[i][0], orderedRects[i][1], orderedRects[i][2], orderedRects[i][3]
		};
		Point2f dst[4] = {
			Point2f(0, 0), Point2f(w - 1, 0), Point2f(w - 1, h - 1), Point2f(0, h - 1)
		};
		Mat trans = getPerspectiveTransform(src, dst);
		warpPerspective(img, warps[i], trans, Point(w, h));
	}
	return warps;
}


void squareImage(int ratio, Mat& img) {
	int H = ((int)img.rows / ratio + 1) * ratio;
	int W = ((int)img.cols / ratio + 1) * ratio;
	int M = img.rows;
	int N = img.cols;
	int HDiff = H - M;
	int WDiff = W - N;
	int HL = 0, HR = 0, WL = 0, WR = 0;
	if (HDiff % 2 == 0) {
		HL = HR = HDiff / 2;
	}
	else {
		HL = (int)HDiff / 2 + 1;
		HR = (int)HDiff / 2;
	}
	if (WDiff % 2 == 0) {
		WL = WR = WDiff / 2;
	}
	else {
		WL = (int)WDiff / 2 + 1;
		WR = (int)WDiff / 2;
	}	
	copyMakeBorder(img, img, HL, HR, WL, WR, BORDER_CONSTANT, 255);
}


vector<Mat> tile(Mat warp) {
	int ratio = 7;
	squareImage(ratio, warp);
	int H = warp.rows;
	int W = warp.cols;
	int M = H / ratio;
	int N = W / ratio;
	int ignore_pixs = IGNORE_MARGIN * min(M, N);
	vector<Mat> grids;
	for (int y(0); y < H; y += M) {
		for (int x(0); x < W; x += N) {
			grids.push_back(warp(Rect(x + ignore_pixs, y + ignore_pixs, N - 2 * ignore_pixs, M - 2 * ignore_pixs)));
		}
	}
	return grids;
}
