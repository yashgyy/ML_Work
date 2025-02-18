// Updated client.cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include "data_loader.cpp" // Include the data loader

using namespace Eigen;
using boost::asio::ip::tcp;

const double LEARNING_RATE = 0.01;
const int MAX_EPOCHS = 3;
const int TRAIN_BATCH_SIZE = 100;
const int NETWORK_BATCH_SIZE = 100;

// Sigmoid function
double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
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

bool train_incrementally(const MatrixXd& data, const VectorXd& labels, VectorXd& weights, double learning_rate, tcp::socket& socket) {
    int n_samples = data.rows();
    int n_features = data.cols();
    static int last_sample = 0;

    VectorXd gradient = VectorXd::Zero(n_features);
    int batch_count = 0;

    std::cout << "[DEBUG] Resuming training from sample index: " << last_sample << std::endl;

    for (int i = last_sample; i < n_samples; ++i) {
        VectorXd xi = data.row(i);
        double yi = labels(i);
        gradient += xi * (sigmoid(xi.dot(weights)) - yi);
        batch_count++;

        if (batch_count == TRAIN_BATCH_SIZE || i == n_samples - 1) {
            weights -= learning_rate * gradient / batch_count;
            std::cout << "[DEBUG] Sending weights after batch." << std::endl;
            send_in_batches(socket, weights);

            gradient.setZero();
            batch_count = 0;

            if (i == n_samples - 1) {
                last_sample = 0;
                return true;
            }

            last_sample = i + 1;
            return false;
        }
    }
    return true;
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::socket socket(io_context);
        socket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 8080));

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

        std::cout << "[INFO] Data loaded and converted." << std::endl;

        // Send vector size to the server
        int vector_size = local_data.cols();
        std::cout << "[DEBUG] Sending vector size: " << vector_size << std::endl;
        boost::asio::write(socket, boost::asio::buffer(&vector_size, sizeof(int)));

        VectorXd weights = VectorXd::Zero(local_data.cols());

        for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
            std::cout << "[INFO] Starting epoch " << epoch + 1 << std::endl;
            while (!train_incrementally(local_data, local_labels, weights, LEARNING_RATE, socket)) {
                // Continue training in incremental batches
            }
        }

        socket.close();

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in client: " << e.what() << std::endl;
    }

    return 0;
}
