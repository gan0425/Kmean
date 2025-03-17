#include "BilateralFilter.h"

/**
 * @brief Converts a 2D double array to OpenCV Mat format
 *
 * @param data Input 2D array of doubles
 * @param rows Number of rows in the array
 * @param cols Number of columns in the array
 * @return cv::Mat OpenCV matrix in CV_64F format (double precision)
 *
 * @note Memory handling:
 * - Creates a new cv::Mat object
 * - Performs deep copy of data
 * - Original array remains unchanged
 */
cv::Mat BilateralFilter::convertToMat(double** data, int rows, int cols) {
    // Initialize OpenCV matrix with double precision
    cv::Mat mat(rows, cols, CV_64F); // CV_64F = 64-bit floating point (double)

    // Copy data row by row using memcpy for efficiency
    for (int i = 0; i < rows; i++) {
        double* matRow = mat.ptr<double>(i);  // Get pointer to mat row
        memcpy(matRow, data[i], cols * sizeof(double));  // Fast memory copy
    }
    return mat;  // Returns by value, but OpenCV handles copying efficiently
}

/**
 * @brief Converts OpenCV Mat back to 2D double array
 *
 * @param mat Input OpenCV matrix (must be CV_64F type)
 * @param data Output 2D array (must be pre-allocated)
 *
 * @warning data array must be pre-allocated with correct dimensions
 * @note Performs direct memory copy for efficiency
 */
void BilateralFilter::convertToArray(const cv::Mat& mat, double** data) {
    // Copy data back row by row
    for (int i = 0; i < mat.rows; i++) {
        const double* matRow = mat.ptr<double>(i);  // Get const pointer to mat row
        memcpy(data[i], matRow, mat.cols * sizeof(double));  // Fast memory copy
    }
}

/**
 * @brief Applies bilateral filter to image data in place
 *
 * @param data Input/output 2D array (modified in place)
 * @param rows Number of rows in the array
 * @param cols Number of columns in the array
 * @param spatialSigma Spatial sigma parameter (σs)
 *        - Controls how much distant pixels influence the result
 *        - Larger values mean more spatial smoothing
 *        - Typical range: 1.0 to 16.0
 * @param intensitySigma Intensity sigma parameter (σr)
 *        - Controls how much pixels with different intensities influence each other
 *        - Larger values allow more intensity mixing
 *        - Typical range: 10.0 to 50.0
 * @param d Diameter of pixel neighborhood
 *        - Must be odd number
 *        - Larger values = more pixels considered = slower processing
 *        - Typical range: 3 to 15
 *
 * @throws cv::Exception if OpenCV operations fail
 *
 * @note Implementation steps:
 * 1. Converts input to OpenCV Mat format (64-bit double)
 * 2. Converts to 32-bit float for bilateral filter
 * 3. Applies bilateral filter
 * 4. Converts back to double precision
 * 5. Copies result back to input array
 */
void BilateralFilter::filterInPlace(
    double** data,           // Input/output array
    int rows,                // Image height
    int cols,                // Image width
    double spatialSigma,     // Spatial sigma (σs)
    double intensitySigma,   // Intensity sigma (σr)
    int d) {                 // Kernel diameter
    try {
        // Step 1: Convert input array to cv::Mat (CV_64F)
        cv::Mat inputMat = convertToMat(data, rows, cols);

        // Step 2: Convert to 32-bit float
        // OpenCV bilateral filter works best with float type
        cv::Mat floatMat;
        inputMat.convertTo(floatMat, CV_32F);

        // Step 3: Apply bilateral filter
        // Parameters:
        // - d: Diameter of pixel neighborhood
        // - intensitySigma: Filter sigma in color space
        // - spatialSigma: Filter sigma in coordinate space
        cv::Mat filteredMat;
        cv::bilateralFilter(floatMat, filteredMat, d, intensitySigma, spatialSigma);

        // Step 4: Convert back to double precision
        cv::Mat outputMat;
        filteredMat.convertTo(outputMat, CV_64F);

        // Step 5: Copy result back to the original array
        convertToArray(outputMat, data);
    }
    catch (const cv::Exception& e) {
        // Log OpenCV errors and rethrow
        std::cerr << "OpenCV error: " << e.what() << std::endl;
        throw;  // Rethrow to allow caller to handle error
    }
}