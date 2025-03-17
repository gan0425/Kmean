
#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <cfloat>
using namespace std;
#pragma once
class kmeans {
public:
    kmeans();
    void setInitialCentroids(const std::vector<double>& values) {
        centroidValues = values;
        hasInitialCentroids = true;
        initialCentroids = values;
    }

    // Initialize with image data
    void setImageData(double** data, int rows, int columns);

    // Set clustering parameters
    void setClusterCount(int count) { maxClusters = count; }
    void setMaxIterations(int iterations) { maxIterations = iterations; }

    // Perform clustering
    void performClustering();

    // Get results
    int** getPixelClusters() const { return pixelClusters; }
    std::vector<double> getMeanIntensities() const;
    std::vector<double> getMinIntensities() const;
    std::vector<int> getPixelCounts() const;
    std::vector<double> getMaxIntensities() const;
    // Add this to the kmeans.h file in the public section:
    int getClusterIndex(int row, int col) const;
    // Add this to the kmeans.h file in the public section:
  
    // Statistics structures
    struct ClusterStats {
        double intensitySum = 0.0;
        int count = 0;
        double minIntensity = DBL_MAX;
        double maxIntensity = -DBL_MAX;
    };

    // Get statistics
        // Cluster statistics
    
    const std::vector<ClusterStats>& getClusterStats() const { return clusterStats; }


private:
    void initializeCentroids(double** inputImage, std::vector<double>& clusterCenters, int K);

    void kMeansClusteringFull(double** inputImage, int**& pixelClusters,
        std::vector<double>& clusterCenters, int K, int maxIterations);
    std::vector<double> initialCentroids;
    bool hasInitialCentroids = false;
    void calculateStatistics();
void selectFixedInitialCentroids();
private:
    double** imageData = nullptr; // Pointer to image data
    int** pixelClusters = nullptr; // Cluster assignments
    std::vector<ClusterStats> clusterStats;
    int maxClusters;
    int maxIterations;
    int currentHeight;
    int currentWidth;
    std::vector<double> centroidValues; // Add this line

};

#endif // KMEANS_H


