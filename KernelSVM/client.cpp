#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include "data_loader.cpp"
//#include "data_loader_susy.cpp"
//#include "data_loader_higgs.cpp"

using namespace Eigen;
using boost::asio::ip::tcp;

const int MAX_EPOCHS = 3;
const double EPSILON = 1e-5;
const int TRAIN_BATCH_SIZE = 30;
const int NETWORK_BATCH_SIZE = 30;

double rbf_kernel(const VectorXd& x1, const VectorXd& x2, double gamma = 0.1) {
    return std::exp(-gamma * (x1 - x2).squaredNorm());
}

void send_in_batches(tcp::socket& socket, const VectorXd& data) {
    int total_size = data.size();
    for (int i = 0; i < total_size; i += NETWORK_BATCH_SIZE) {
        int batch_size = std::min(NETWORK_BATCH_SIZE, total_size - i);
        boost::asio::write(socket, boost::asio::buffer(data.data() + i, batch_size * sizeof(double)));
    }
}

bool train_incrementally(const MatrixXd& data, const VectorXd& labels,
                         VectorXd& weights, double learning_rate,
                         double gamma, tcp::socket& socket) {
    //std::cout<<"Inside Function 1"<<std::endl;
    int n_samples = data.rows();
    static int last_sample = 0;
    VectorXd gradient = VectorXd::Zero(weights.size());
    int current_sample = last_sample;
    int processed_samples = 0;
    int Iteration = n_samples / TRAIN_BATCH_SIZE;

    while (processed_samples < Iteration) {
        if (current_sample >= n_samples) return true; // All samples processed
        std::cout<<current_sample<<std::endl;

        VectorXd xi = data.row(current_sample);
        double yi = labels(current_sample);
        double kernel_output = 0.0;

        for (int j = 0; j < weights.size(); ++j)
            kernel_output += rbf_kernel(xi, data.row(j), gamma) * weights(j);

        if (yi * kernel_output < 1) {
            for (int j = 0; j < weights.size(); ++j)
                gradient(j) += -yi * rbf_kernel(data.row(j), xi, gamma);
        }

        weights -= learning_rate * gradient;
        current_sample++;
        processed_samples++;

        if (processed_samples % TRAIN_BATCH_SIZE == 0) {
            int batch_size = TRAIN_BATCH_SIZE;
            int vector_size = weights.size();
            boost::asio::write(socket, boost::asio::buffer(&batch_size, sizeof(int)));
            boost::asio::write(socket, boost::asio::buffer(&vector_size, sizeof(int)));
            send_in_batches(socket, weights);
            boost::asio::read(socket, boost::asio::buffer(weights.data(), weights.size() * sizeof(double)));
            std::cout<<"Send and Recieved"<<std::endl;
        }
    }

    last_sample = current_sample;
    return false;
}

int main() {
    try {
        std::vector<std::vector<float>> features;
        std::vector<int> labels;
        load_data("../Datasets/santander-customer-transaction-prediction.csv", features, labels);
        //load_data("../Datasets/SUSY.csv", features, labels);
        //load_data("../Datasets/HIGGS.csv", features, labels);

        MatrixXd local_data(features.size(), features[0].size());
        VectorXd local_labels(labels.size());
        for (size_t i = 0; i < features.size(); ++i) {
            for (size_t j = 0; j < features[i].size(); ++j)
                local_data(i, j) = features[i][j];
            local_labels(i) = labels[i];
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.01);
        VectorXd local_weights = VectorXd::Zero(local_data.rows()).unaryExpr([&](double) { return d(gen); });

        double learning_rate = 0.01;
        double gamma = 0.1;

        boost::asio::io_context io_context;
        tcp::socket socket(io_context);
        std::cout << "[DEBUG] Connecting to server..." << std::endl;
        socket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 8080));
        std::cout << "[DEBUG] Connected to server." << std::endl;

        // int vector_size = local_weights.size();
        // boost::asio::write(socket, boost::asio::buffer(&vector_size, sizeof(int)));

        for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
            bool exit = train_incrementally(local_data, local_labels, local_weights, learning_rate, gamma, socket);
            if (exit) break;
        }

        socket.close();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in client: " << e.what() << std::endl;
    }

    return 0;
}
