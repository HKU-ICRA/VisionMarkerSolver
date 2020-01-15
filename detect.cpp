#include <cmath>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "detect.h"
#include "markerDict.h"


using namespace std;
using namespace cv;


/* Tuneable variables */
//Scalar mcolor_lb(15, 100, 150);
//Scalar mcolor_ub(30, 255, 255);
Scalar mcolor_lb(10, 50, 100);
Scalar mcolor_ub(30, 255, 255);
double COVERAGE = 0.8;


/* Constructors */
vector<vector<Point> > orderRects(vector<vector<Point> >& rects);
vector<Mat> perspectiveTrans(Mat img, vector<vector<Point> >& orderedRects);
vector<Mat> tile(Mat warp);


/* dictionary of markers */
markerDict dictionary;


void detectMarker(Mat img, vector<string>& marker_names, vector<vector<Point> >& marker_rects) {
	Mat img_og = img.clone();
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
	//drawContours(img_og, rects, 0, Scalar(0, 255, 0), 3);
	//imshow("Img og", img_og);
	//waitKey(0);
	/* Prune rectangles by removing those within existing rectangles */
	// TO-DO
	/* Perspective transform */
	vector<vector<Point> > orderedRects = orderRects(rects);
	vector<Mat> warps = perspectiveTrans(img, orderedRects);
	/* Split image into 7x7 tiles */
	vector<vector<Mat> > grids;
	int deleted_boxes = 0;
	for (int i(0); i < warps.size(); ++i) {
		if (warps[i].rows >= 7 && warps[i].cols >= 7) {
			grids.push_back(tile(warps[i]));
		}
		else {
			orderedRects.erase(orderedRects.begin() + i - deleted_boxes);
			++deleted_boxes;
		}
;	}
	/* Better padding / cropping to preserve symmetry */
	// TO DO
	/* Determine whether cell is filled and return detected markers */
	for (int i(0); i < grids.size(); ++i) {
		vector<int> marker(49);
		for (int g(0); g < grids[i].size(); ++g) {
			marker[g] = ((double)countNonZero(grids[i][g]) / 49.0) >= COVERAGE;
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
		/*
		vector<Point> dst {
			Point(0, 0), Point(w - 1, 0), Point(w - 1, h - 1), Point(0, h - 1)
		};
		*/
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
	vector<Mat> grids;
	for (int y(0); y < H; y += M) {
		for (int x(0); x < W; x += N) {
			grids.push_back(warp(Rect(x, y, N, M)));
		}
	}
	return grids;
}
