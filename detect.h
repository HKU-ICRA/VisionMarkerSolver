#ifndef DETECT_H
#define DETECT_H


#include <vector>
#include <string>

#include <opencv2/core.hpp>


void detectMarker(cv::Mat img, std::vector<std::string>& marker_names, std::vector<std::vector<cv::Point> >& marker_rects);


#endif
