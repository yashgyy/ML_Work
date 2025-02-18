#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include "data_loader.cpp"  // Include the data loader

using namespace Eigen;
using boost::asio::ip::tcp;

const int MAX_EPOCHS = 3;
const double EPSILON = 1e-5;
const int TRAIN_BATCH_SIZE = 100;
const int NETWORK_BATCH_SIZE = 100;

void send_in_batches(tcp::socket& socket, const VectorXd& data) {
    int total_size = data.size();
    std::cout << "[INFO] Sending data in batches of size: " << NETWORK_BATCH_SIZE << std::endl;

    for (int i = 0; i < total_size; i += NETWORK_BATCH_SIZE) {
        int batch_size = std::min(NETWORK_BATCH_SIZE, total_size - i);
        std::cout << "[DEBUG] Sending batch " << (i / NETWORK_BATCH_SIZE) + 1
                  << " of size: " << batch_size << std::endl;

        boost::asio::write(socket, boost::asio::buffer(data.data() + i, batch_size * sizeof(double)));
    }
}

bool train_incrementally(const MatrixXd& data, const VectorXd& labels, VectorXd& weights, tcp::socket& socket) {
    int n_samples = data.rows();
    static int last_sample = 0;

    VectorXd gradient = VectorXd::Zero(weights.size());
    int processed_samples = 0;
    int current_sample = last_sample;

    std::cout << "[DEBUG] Starting training from sample: " << last_sample << std::endl;

    while (processed_samples < TRAIN_BATCH_SIZE) {
        if (current_sample >= n_samples) {
            std::cout << "[INFO] All samples processed. Exiting program." << std::endl;
            return true;
        }

        VectorXd xi = data.row(current_sample);
        double yi = labels(current_sample);

        if (yi * (xi.dot(weights)) < 1) {
            gradient += -yi * xi;
        }

        weights -= 0.01 * gradient;

        current_sample++;
        processed_samples++;

        if (processed_samples % TRAIN_BATCH_SIZE == 0) {
            std::cout << "[DEBUG] Sending weights after processing " << TRAIN_BATCH_SIZE << " samples." << std::endl;
            send_in_batches(socket, weights);
        }
    }

    last_sample = current_sample;
    return false;
}

int main() {
    try {
        std::vector<std::vector<float>> features;
        std::vector<int> labels;
        load_data("train.csv", features, labels);

        MatrixXd data(features.size(), features[0].size());
        VectorXd label_vec(labels.size());
        for (size_t i = 0; i < features.size(); ++i) {
            for (size_t j = 0; j < features[i].size(); ++j) {
                data(i, j) = features[i][j];
            }
            label_vec(i) = labels[i];
        }

        VectorXd weights = VectorXd::Random(data.cols());
        boost::asio::io_context io_context;
        tcp::socket socket(io_context);
        socket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 8080));

        int vector_size = weights.size();
        std::cout << "[DEBUG] Sending vector size: " << vector_size << std::endl;
        boost::asio::write(socket, boost::asio::buffer(&vector_size, sizeof(int)));
        boost::asio::write(socket, boost::asio::buffer(weights.data(), weights.size() * sizeof(double)));

        for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
            std::cout << "[INFO] Starting epoch " << epoch + 1 << std::endl;

            if (train_incrementally(data, label_vec, weights, socket)) {
                break;
            }
        }

        socket.close();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in client: " << e.what() << std::endl;
    }

    return 0;
}
