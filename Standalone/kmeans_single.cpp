// Pure KMeans Implementation - No networking, no federated logic
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <Eigen/Dense>
#include <chrono>

using namespace Eigen;

class KMeans {
private:
    int k;
    int max_iterations;
    double tolerance;
    
public:
    KMeans(int clusters = 3, int max_iters = 150, double tol = 1e-6)
        : k(clusters), max_iterations(max_iters), tolerance(tol) {}

    // Initialize centroids by randomly selecting k points from data
    MatrixXd initialize_centroids(const MatrixXd& data) {
        std::vector<int> indices(data.rows());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        MatrixXd centroids(k, data.cols());
        for (int i = 0; i < k; ++i) {
            centroids.row(i) = data.row(indices[i]);
        }
        return centroids;
    }

    // Perform one iteration of KMeans
    MatrixXd kmeans_single_iter(const MatrixXd& data, const MatrixXd& centroids) {
        int n_samples = data.rows();
        int k = centroids.rows();
        VectorXi labels(n_samples);

        // Assign each point to nearest centroid
        for (int i = 0; i < n_samples; ++i) {
            RowVectorXd point = data.row(i);
            VectorXd distances = (centroids.rowwise() - point).rowwise().squaredNorm();
            distances.minCoeff(&labels(i));
        }

        // Update centroids
        MatrixXd new_centroids = MatrixXd::Zero(k, data.cols());
        std::vector<int> counts(k, 0);

        for (int i = 0; i < n_samples; ++i) {
            new_centroids.row(labels(i)) += data.row(i);
            counts[labels(i)]++;
        }

        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0)
                new_centroids.row(j) /= counts[j];
        }

        return new_centroids;
    }

    // Main clustering function
    MatrixXd fit(const MatrixXd& data) {
        MatrixXd centroids = initialize_centroids(data);
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            MatrixXd new_centroids = kmeans_single_iter(data, centroids);
            
            // Check convergence
            if ((new_centroids - centroids).norm() < tolerance) {
                std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
                return new_centroids;
            }
            
            centroids = new_centroids;
        }
        
        return centroids;
    }

    // Get cluster assignments for data points
    VectorXi get_labels(const MatrixXd& data, const MatrixXd& centroids) {
        VectorXi labels(data.rows());
        for (int i = 0; i < data.rows(); ++i) {
            RowVectorXd point = data.row(i);
            VectorXd distances = (centroids.rowwise() - point).rowwise().squaredNorm();
            distances.minCoeff(&labels(i));
        }
        return labels;
    }
};

// Simple data generator for testing
MatrixXd generate_test_data(int n_samples = 100) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::normal_distribution<> dist(0.0, 1.0);
    
    MatrixXd data(n_samples, 2);
    for (int i = 0; i < n_samples; ++i) {
        data(i, 0) = dist(g);
        data(i, 1) = dist(g);
    }
    return data;
}

int main() {
    // Generate sample data
    MatrixXd data = generate_test_data(200);
    
    // Create and run KMeans
    KMeans kmeans(3, 100, 1e-6);
    MatrixXd centroids = kmeans.fit(data);
    
    std::cout << "Final centroids:\n" << centroids << std::endl;
    
    // Get cluster assignments
    VectorXi labels = kmeans.get_labels(data, centroids);
    std::cout << "First 10 cluster assignments: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << labels(i) << " ";
    }
    std::cout << std::endl;
    
    return 0;
}