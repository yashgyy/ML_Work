// Refactored Federated KMeans Client (Based on K_client structure)
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include <numeric>
#include "data_loader.cpp"

using namespace Eigen;
using boost::asio::ip::tcp;

const int K = 2; // number of clusters
const int MAX_ITERS = 100;
const int NETWORK_BATCH_SIZE = 100;

// Function to simulate local data
MatrixXd load_local_data() {
    MatrixXd data(4, 2);
    data << 1.0, 2.0,
            2.0, 1.0,
            3.0, 4.0,
            4.0, 3.0;
    return data;
}

// Random initialization of centroids
MatrixXd initialize_centroids(const MatrixXd& data, int k) {
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

// One iteration of local KMeans
MatrixXd kmeans_single_iter(const MatrixXd& data, const MatrixXd& centroids) {
    int n_samples = data.rows();
    int k = centroids.rows();
    VectorXi labels(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        RowVectorXd point = data.row(i);
        VectorXd distances = (centroids.rowwise() - point).rowwise().squaredNorm();
        distances.minCoeff(&labels(i));
    }

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

void send_matrix(tcp::socket& socket, const MatrixXd& matrix) {
    int rows = matrix.rows(), cols = matrix.cols();
    boost::asio::write(socket, boost::asio::buffer(&rows, sizeof(int)));
    boost::asio::write(socket, boost::asio::buffer(&cols, sizeof(int)));
    boost::asio::write(socket, boost::asio::buffer(matrix.data(), rows * cols * sizeof(double)));
}

void receive_matrix(tcp::socket& socket, MatrixXd& matrix) {
    int rows, cols;
    boost::asio::read(socket, boost::asio::buffer(&rows, sizeof(int)));
    boost::asio::read(socket, boost::asio::buffer(&cols, sizeof(int)));
    matrix.resize(rows, cols);
    boost::asio::read(socket, boost::asio::buffer(matrix.data(), rows * cols * sizeof(double)));
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::socket socket(io_context);
        socket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 8080));
        
        std::vector<float> features;
        std::vector<int> labels;
       
        load_data("../Datasets/circles.csv", features, labels);
        // Convert to Eigen matrices
        MatrixXd local_data(features.size(), 2);
        for (size_t i = 0; i < features.size(); ++i) {
                local_data(i, 0) = features[i];
                local_data(i, 1) = labels[i];
            }

        //data = local_data;
        MatrixXd centroids = initialize_centroids(local_data, K);


        for (int iter = 0; iter < MAX_ITERS; ++iter) {
            MatrixXd local_centroids = kmeans_single_iter(local_data, centroids);

            // Send local centroids
            send_matrix(socket, local_centroids);

            // Receive updated global centroids
            receive_matrix(socket, centroids);

            std::cout << "[INFO] Iteration " << iter + 1 << " updated centroids:\n" << centroids << std::endl;
        }
        socket.close();

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in client: " << e.what() << std::endl;
    }

    return 0;
}