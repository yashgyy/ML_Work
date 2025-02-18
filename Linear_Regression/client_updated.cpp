// Updated client.cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include "data_loader.cpp"  // Include the data loader

using namespace Eigen;
using boost::asio::ip::tcp;

const int MAX_EPOCHS = 3;       // Maximum number of epochs
const double EPSILON = 1e-5;      // Tolerance for convergence
const int TRAIN_BATCH_SIZE = 100; // Batch size for incremental training
const int NETWORK_BATCH_SIZE = 100; // Batch size for network transmission

void normalize_data(MatrixXd& data) {
    for (int j = 0; j < data.cols(); ++j) {
        double mean = data.col(j).mean();
        double stddev = std::sqrt((data.col(j).array() - mean).square().mean());
        data.col(j) = (data.col(j).array() - mean) / stddev;
    }
}

void send_in_batches(tcp::socket& socket, const VectorXd& data) {
    int total_size = data.size();
    std::cout << "[INFO] Sending data in batches of size: " << NETWORK_BATCH_SIZE << std::endl;

    int sent = 0;
    while (sent < total_size) {
        int batch_size = std::min(NETWORK_BATCH_SIZE, total_size - sent);
        boost::asio::write(socket, boost::asio::buffer(data.data() + sent, batch_size * sizeof(double)));
        sent += batch_size;
        std::cout << "[DEBUG] Sent batch of size: " << batch_size << std::endl;
    }
}

void clip_gradients(VectorXd& gradient, double max_norm) {
    double norm = gradient.norm();
    max_norm = std::min(max_norm, 10.0); // Ensure an upper cap on max_norm
    if (norm > max_norm) {
        gradient *= (max_norm / norm);
    }
}

bool train_incrementally(const MatrixXd& data, const VectorXd& labels, 
                         VectorXd& weights, double learning_rate, 
                         tcp::socket& socket) {
    int n_samples = data.rows();
    int n_features = data.cols();
    static int last_sample = 0;  // Remember the last sample across function calls

    VectorXd gradient = VectorXd::Zero(n_features);
    int batch_count = 0;

    std::cout << "[DEBUG] Resuming training from sample index: " << last_sample << std::endl;

    for (int i = last_sample; i < n_samples; ++i) {
        VectorXd xi = data.row(i);
        double yi = labels(i);
        gradient += -2 * xi * (yi - xi.dot(weights));
        batch_count++;

        if (batch_count == TRAIN_BATCH_SIZE || i == n_samples - 1) {
            clip_gradients(gradient, 1.0); // Clip gradients to a max norm of 1.0

            if (!gradient.allFinite()) {
                std::cerr << "[ERROR] Gradient contains NaN or inf values. Resetting gradient." << std::endl;
                gradient.setZero(); // Reset gradient to prevent invalid updates
            }

            weights -= learning_rate * gradient / batch_count;

            if (!weights.allFinite()) {
                std::cerr << "[ERROR] Weights contain NaN or inf values." << std::endl;
                weights.setZero(); // Reset weights to prevent cascading errors
            }

            std::cout << "[DEBUG] Sending weights after batch." << std::endl;
            send_in_batches(socket, weights);

            gradient.setZero();  // Reset gradient for the next batch
            batch_count = 0;

            if (i == n_samples - 1) {
                last_sample = 0;  // Reset for the next epoch
                return true;  // Completed one epoch
            }

            last_sample = i + 1;  // Update the state for the next call
            return false;  // Continue training
        }
    }
    return true;
}

int main() {
    try {
        // Load data from train.csv
        std::vector<std::vector<float>> features;
        std::vector<int> labels;
        load_data("/home/yash/Work/Profiling_Fed/ML/KernelSVM/train.csv", features, labels);

        // Convert to Eigen matrices
        MatrixXd local_data(features.size(), features[0].size());
        VectorXd local_labels(labels.size());
        for (size_t i = 0; i < features.size(); ++i) {
            for (size_t j = 0; j < features[i].size(); ++j) {
                local_data(i, j) = features[i][j];
            }
            local_labels(i) = labels[i];
        }

        normalize_data(local_data); // Normalize the data
        std::cout << "[INFO] Normalization complete." << std::endl;

        // Initialize weights with small random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.01);
        VectorXd local_weights = VectorXd::Zero(local_data.cols()).unaryExpr([&](double) { return d(gen); });

        double learning_rate = 1e-4; // Reduced learning rate

        // Connect to the server
        boost::asio::io_context io_context;
        tcp::socket socket(io_context);
        socket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 8080));

        // Send vector size to the server
        int vector_size = local_weights.size();
        std::cout << "[DEBUG] Sending vector size: " << vector_size << std::endl;
        boost::asio::write(socket, boost::asio::buffer(&vector_size, sizeof(int)));

        // Train incrementally in batches
        for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
            std::cout << "[INFO] Starting epoch " << epoch + 1 << std::endl;

            while (!train_incrementally(local_data, local_labels, local_weights, learning_rate, socket)) {
                // Continue training in incremental batches
            }
        }

        socket.close();

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in client: " << e.what() << std::endl;
    }

    return 0;
}
