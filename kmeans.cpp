#include "kmeans.h"
#include <set>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <thread>
#include <future>
#include <xmmintrin.h> 


// Structure to hold cluster information for sorting
struct ClusterInfo {
    int originalIndex;
    double meanIntensity;
    std::vector<std::pair<int, int>> pixels; // (y,x) coordinates
    double intensity;
};
/**
 * @brief Default constructor for the kmeans class
 * No initialization needed as member variables are initialized when setting image data
 */
kmeans::kmeans() {}

/**
 * @brief Sets the input image data for clustering
 * @param data Pointer to 2D array containing image intensity values
 * @param rows Number of rows in the image
 * @param columns Number of columns in the image
 *
 * This should be called first before performing any clustering operations
 */
void kmeans::setImageData(double** data, int rows, int columns) {
    imageData = data;
    currentHeight = rows;
    currentWidth = columns;
}

/**
 * @brief Initializes cluster centroids either from user-provided values or automatically
 * @param inputImage Input image data array
 * @param clusterCenters Vector to store the initialized centroids
 * @param K Number of clusters
 *
 * If hasInitialCentroids is true, uses pre-set centroids from initialCentroids
 * Otherwise, automatically selects centroids by evenly distributing them across the intensity range
 *
 * @throws std::invalid_argument if input parameters are invalid
 * @throws std::runtime_error if centroid initialization fails
 */




 // Add a new method for fixed centroid selection
void kmeans::selectFixedInitialCentroids() {
    // Method to select initial centroids automatically and deterministically
    hasInitialCentroids = true;
    initialCentroids.clear();

    // Example: Select centroids based on evenly distributed intensity values
    double minIntensity = std::numeric_limits<double>::max();
    double maxIntensity = std::numeric_limits<double>::lowest();

    // Find min and max intensities
    for (int y = 0; y < currentHeight; ++y) {
        for (int x = 0; x < currentWidth; ++x) {
            if (!std::isnan(imageData[y][x]) && !std::isinf(imageData[y][x])) {
                minIntensity = std::min(minIntensity, imageData[y][x]);
                maxIntensity = std::max(maxIntensity, imageData[y][x]);
            }
        }
    }

    // Generate initial centroids
    for (int i = 0; i < maxClusters; ++i) {
        double centroidValue = minIntensity +
            (maxIntensity - minIntensity) * (i + 1.0) / (maxClusters + 1.0);
        initialCentroids.push_back(centroidValue);
    }
}





void kmeans::initializeCentroids(double** inputImage, std::vector<double>& clusterCenters, int K) {
    // Input validation
    if (!inputImage || currentHeight <= 0 || currentWidth <= 0) {
        throw std::invalid_argument("Invalid input image or dimensions");
    }
    if (K <= 0) {
        throw std::invalid_argument("Number of clusters (K) must be positive");
    }

    // If initial centroids are provided by mouse selection, use a deterministic method
    if (!centroidValues.empty()) {
        // Sort the mouse-selected values to ensure consistent ordering
        std::vector<double> sortedCentroidValues = centroidValues;
        std::sort(sortedCentroidValues.begin(), sortedCentroidValues.end());

        clusterCenters = sortedCentroidValues;

        // If fewer values than required clusters, repeat the values
        while (clusterCenters.size() < K) {
            clusterCenters.push_back(sortedCentroidValues[clusterCenters.size() % sortedCentroidValues.size()]);
        }

        // If more values than required clusters, truncate
        if (clusterCenters.size() > K) {
            clusterCenters.resize(K);
        }

        return;
    }

    // Existing approach for automatic centroid selection
    // Collect unique pixel values
    std::set<double> uniqueValues;
    for (int y = 0; y < currentHeight; ++y) {
        for (int x = 0; x < currentWidth; ++x) {
            if (!std::isnan(inputImage[y][x]) && !std::isinf(inputImage[y][x])) {
                uniqueValues.insert(inputImage[y][x]);
            }
        }
    }

    if (uniqueValues.empty()) {
        throw std::runtime_error("No valid pixel values found in the image");
    }

    // Convert set to vector for indexed access
    std::vector<double> sortedValues(uniqueValues.begin(), uniqueValues.end());
    std::sort(sortedValues.begin(), sortedValues.end());

    int uniqueCount = sortedValues.size();
    clusterCenters.resize(K);

    // Distribute centroids evenly
    if (uniqueCount >= K) {
        for (int i = 0; i < K; ++i) {
            double fraction = static_cast<double>(i) / (K - 1);
            int index = static_cast<int>(fraction * (uniqueCount - 1));
            clusterCenters[i] = sortedValues[index];
        }
    }
    else {
        // Handle case with fewer unique values than clusters
        for (int i = 0; i < K; ++i) {
            clusterCenters[i] = sortedValues[i % uniqueCount];
        }
    }

    // Verify centroid initialization
    for (int i = 0; i < K; ++i) {
        if (std::isnan(clusterCenters[i]) || std::isinf(clusterCenters[i])) {
            throw std::runtime_error("Failed to initialize valid centroid values");
        }
    }
}

/**
 * @brief Performs k-means clustering using parallel processing
 * @param inputImage Input image data array
 * @param pixelClusters Output array for cluster assignments
 * @param clusterCenters Vector of cluster centroids
 * @param K Number of clusters
 * @param maxIterations Maximum number of iterations
 *calculation is based on the pixel value (pixelValue) and the current cluster centers (clusterCenters).
 * Uses multi-threading to speed up clustering process
 * Automatically determines optimal thread count based on hardware
 */
void kmeans::kMeansClusteringFull(double** inputImage, int**& pixelClusters, std::vector<double>& clusterCenters,
    int K, int maxIterations) {

    // Temporary arrays for the clustering process
    std::vector<ClusterInfo> clusterInfos(K);
    for (int k = 0; k < K; k++) {
        clusterInfos[k].originalIndex = k;
        clusterInfos[k].meanIntensity = clusterCenters[k];
    }

    // Initialize pixel clusters array
    pixelClusters = new int* [currentHeight];
    for (int i = 0; i < currentHeight; ++i) {
        pixelClusters[i] = new int[currentWidth];
        std::fill(pixelClusters[i], pixelClusters[i] + currentWidth, 0);
    }

    std::vector<double> newClusterCenters(K);
    std::vector<int> clusterCounts(K);
    const int numThreads = std::thread::hardware_concurrency();
    const int rowsPerThread = currentHeight / numThreads;

    // Main clustering loop
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        std::fill(newClusterCenters.begin(), newClusterCenters.end(), 0.0);
        std::fill(clusterCounts.begin(), clusterCounts.end(), 0);

        std::vector<std::future<std::pair<std::vector<double>, std::vector<int>>>> futures;

        for (int t = 0; t < numThreads; ++t) {
            int startRow = t * rowsPerThread;
            int endRow = (t == numThreads - 1) ? currentHeight : (t + 1) * rowsPerThread;

            futures.push_back(std::async(std::launch::async,
                [=, &clusterCenters]() -> std::pair<std::vector<double>, std::vector<int>> {
                    std::vector<double> localSums(K, 0.0);
                    std::vector<int> localCounts(K, 0);

                    for (int y = startRow; y < endRow; ++y) {
                        for (int x = 0; x < currentWidth; ++x) {
                            double pixelValue = inputImage[y][x];
                            int nearestCluster = 0;
                            double minDistance = std::abs(pixelValue - clusterCenters[0]);

                            for (int k = 1; k < K; ++k) {
                                double distance = std::abs(pixelValue - clusterCenters[k]);
                                if (distance < minDistance) {
                                    nearestCluster = k;
                                    minDistance = distance;
                                }
                            }

                            pixelClusters[y][x] = nearestCluster;
                            localSums[nearestCluster] += pixelValue;
                            localCounts[nearestCluster]++;
                        }
                    }
                    return std::make_pair(localSums, localCounts);
                }));
        }

        for (auto& future : futures) {
            auto result = future.get();
            for (int k = 0; k < K; ++k) {
                newClusterCenters[k] += result.first[k];
                clusterCounts[k] += result.second[k];
            }
        }

        bool converged = true;
        for (int k = 1; k < K; ++k) {
            double newCenter = clusterCounts[k] > 0 ? newClusterCenters[k] / clusterCounts[k] : clusterCenters[k];
            if (std::abs(newCenter - clusterCenters[k]) > 1e-5) {
                converged = false;
            }
            clusterCenters[k] = newCenter;
            clusterInfos[k].meanIntensity = newCenter;
        }

        if (converged) break;
    }
    // Sort clusters by mean intensity
    std::sort(clusterInfos.begin(), clusterInfos.end(),
        [](const ClusterInfo& a, const ClusterInfo& b) {
            return a.meanIntensity < b.meanIntensity;
        });

    // Create mapping from old to new indices
    std::vector<int> indexMapping(K);
    for (int i = 0; i < K; i++) {
        indexMapping[clusterInfos[i].originalIndex] = i;
    }

    // Update pixel cluster assignments with new sorted indices
    for (int y = 0; y < currentHeight; ++y) {
        for (int x = 0; x < currentWidth; ++x) {
            pixelClusters[y][x] = indexMapping[pixelClusters[y][x]];
        }
    }

    // Update cluster centers to sorted order
    std::vector<double> sortedCenters(K);
    for (int i = 0; i < K; i++) {
        sortedCenters[i] = clusterInfos[i].meanIntensity;
    }
    clusterCenters = sortedCenters;
}

//int** pixelClusters;  // 2D array storing cluster assignments for each pixel
/**
 * @brief Main function to perform clustering on the image
 * Coordinates the entire clustering process including initialization and statistics calculation
 * @throws std::runtime_error if image data hasn't been set
 */
void kmeans::performClustering() {
    if (imageData == nullptr) {
        throw std::runtime_error("Image data not set");
    }

    // Initialize and perform clustering
    std::vector<double> clusterCenters;
    initializeCentroids(imageData, clusterCenters, maxClusters);
    kMeansClusteringFull(imageData, pixelClusters, clusterCenters, maxClusters, maxIterations);
    calculateStatistics();
}

/**
 * @brief Structure to hold statistics for each cluster
 * Used internally to track various metrics for each cluster
 */
struct ClusterStatistics {
    int count = 0;                      // Number of pixels in cluster
    double intensitySum = 0.0;          // Sum of intensities for mean calculation
    double minIntensity = std::numeric_limits<double>::max();  // Minimum intensity
    double maxIntensity = std::numeric_limits<double>::lowest(); // Maximum intensity
};

/**
 * @brief Calculates statistics for all clusters
 * Computes mean intensity, pixel counts, and intensity ranges for each cluster
 */
 //std::vector<ClusterStatistics> clusterStats;  // Vector storing statistics for each cluster
void kmeans::calculateStatistics() {
    clusterStats.resize(maxClusters);

    for (int y = 0; y < currentHeight; ++y) {
        for (int x = 0; x < currentWidth; ++x) {
            int cluster = pixelClusters[y][x];
            double intensity = imageData[y][x];

            clusterStats[cluster].intensitySum += intensity;
            clusterStats[cluster].count++;
            clusterStats[cluster].minIntensity = std::min(clusterStats[cluster].minIntensity, intensity);
            clusterStats[cluster].maxIntensity = std::max(clusterStats[cluster].maxIntensity, intensity);
        }
    }
}


// Getter functions for cluster statistics

/**
 * @brief Gets mean intensity values for each cluster
 * @return Vector of mean intensities
 */
std::vector<double> kmeans::getMeanIntensities() const {
    std::vector<double> meanIntensities;
    for (const auto& info : clusterStats) {
        meanIntensities.push_back(info.count > 0 ? info.intensitySum / info.count : 0.0);
    }
    return meanIntensities;
}

/**
 * @brief Gets pixel counts for each cluster
 * @return Vector of pixel counts
 */
std::vector<int> kmeans::getPixelCounts() const {
    std::vector<int> pixelCounts;
    for (const auto& stats : clusterStats) {
        pixelCounts.push_back(stats.count);
    }
    return pixelCounts;
}

/**
 * @brief Gets minimum intensity values for each cluster
 * @return Vector of minimum intensities
 */
std::vector<double> kmeans::getMinIntensities() const {
    std::vector<double> minIntensities;
    for (const auto& stats : clusterStats) {
        minIntensities.push_back(stats.minIntensity);
    }
    return minIntensities;
}

/**
 * @brief Gets maximum intensity values for each cluster
 * @return Vector of maximum intensities
 */
std::vector<double> kmeans::getMaxIntensities() const {
    std::vector<double> maxIntensities;
    for (const auto& stats : clusterStats) {
        maxIntensities.push_back(stats.maxIntensity);
    }
    return maxIntensities;
}
// Add this to the kmeans.cpp file:
/**
 * @brief Gets the cluster index for a specific pixel
 * @param row Row coordinate of the pixel
 * @param col Column coordinate of the pixel
 * @return Cluster index of the specified pixel
 * @throws std::out_of_range if coordinates are invalid
 */
int kmeans::getClusterIndex(int row, int col) const {
    if (row < 0 || row >= currentHeight || col < 0 || col >= currentWidth) {
        throw std::out_of_range("Pixel coordinates out of range");
    }
    if (pixelClusters == nullptr) {
        throw std::runtime_error("Clustering has not been performed yet");
    }
    return pixelClusters[row][col];
}
