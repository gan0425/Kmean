// BilateralFilter.h
#ifndef BILATERAL_FILTER_H
#define BILATERAL_FILTER_H

#include <cmath>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <execution>
#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace std;
class BilateralFilter {
public:
    static  void filterInPlace(double** data, int rows, int cols,
        double spatialSigma,
        double intensitySigma,
        int d);

private:
    static cv::Mat convertToMat(double** data, int rows, int cols);
    static void convertToArray(const cv::Mat& mat, double** data);
};

#endif // BILATERAL_FILTER_