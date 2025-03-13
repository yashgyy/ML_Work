
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include "data_loader.cpp"  // Include the data loader

using namespace Eigen;
using boost::asio::ip::tcp;
// g++ client_updated.cpp -o client  -I /usr/include/eigen3

const int MAX_EPOCHS = 3;       // Maximum number of epochs
const double EPSILON = 1e-5;      // Tolerance for convergence
const int TRAIN_BATCH_SIZE = 100; // Batch size for incremental training
const int NETWORK_BATCH_SIZE = 100; // Batch size for network transmission

double rbf_kernel(const VectorXd& x1, const VectorXd& x2, double gamma = 0.1) {
    return std::exp(-gamma * (x1 - x2).squaredNorm());
}

double compute_loss(const MatrixXd& data, const VectorXd& labels, 
                    const VectorXd& weights, double gamma) {
    double loss = 0.0;
    int n_samples = data.rows();

    for (int i = 0; i < n_samples; ++i) {
        VectorXd xi = data.row(i);
        double yi = labels(i);
        double prediction = 0.0;

        for (int j = 0; j < n_samples; ++j) {
            prediction += rbf_kernel(xi, data.row(j), gamma) * weights(j);
        }
        loss += std::max(0.0, 1 - yi * prediction);
    }
    return loss / n_samples;
}

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

bool train_incrementally(const MatrixXd& data, const VectorXd& labels, 
                         VectorXd& weights, double learning_rate, 
                         double gamma, tcp::socket& socket) {
    int n_samples = data.rows();
    static int last_sample = 0;  // Remember the last sample across function calls

    VectorXd gradient = VectorXd::Zero(weights.size());  // Declare gradient outside the loop

    // Train incrementally starting from the last processed sample
    int current_sample = last_sample;
    int processed_samples = 0;

    std::cout << "[DEBUG] Starting training from sample: " << last_sample << std::endl;

    while (processed_samples < TRAIN_BATCH_SIZE) {
        if (current_sample >= n_samples) {
            std::cout << "[INFO] All samples processed. Exiting program." << std::endl;
            return true;  // Exit condition: All samples processed
        }

        // Compute the gradient for the current sample
        VectorXd xi = data.row(current_sample);
        double yi = labels(current_sample);
        double kernel_output = 0.0;

        for (int j = 0; j < weights.size(); ++j) {
            kernel_output += rbf_kernel(xi, data.row(j), gamma) * weights(j);
        }

        if (yi * kernel_output < 1) {
            for (int j = 0; j < weights.size(); ++j) {
                gradient(j) += -yi * rbf_kernel(data.row(j), xi, gamma);
            }
        }

        // Update weights after processing each sample
        weights -= learning_rate * gradient;

        // Move to the next sample
        current_sample++;
        processed_samples++;

        // Send weights to server after processing the entire batch
        if (processed_samples % TRAIN_BATCH_SIZE == 0) {
            std::cout << "[DEBUG] Sending weights after processing " << TRAIN_BATCH_SIZE << " samples." << std::endl;
            send_in_batches(socket, weights);
        }
    }

    // Update the last sample index for the next call
    last_sample = current_sample;
    return false;  // Continue training
}

int main() {
    try {
        // Load data from train.csv
        std::vector<std::vector<float>> features;
        std::vector<int> labels;
        //load_data("../Datasets/santander-customer-transaction-prediction.csv", features, labels);
        load_data("../Datasets/santander-customer-transaction-prediction.csv", features, labels);
        // Convert to Eigen matrices
        MatrixXd local_data(features.size(), features[0].size());
        VectorXd local_labels(labels.size());
        for (size_t i = 0; i < features.size(); ++i) {
            for (size_t j = 0; j < features[i].size(); ++j) {
                local_data(i, j) = features[i][j];
            }
            local_labels(i) = labels[i];
        }

        std::cout << "Conversion Done" << std::endl;

        // Initialize weights with small random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.01);
        VectorXd local_weights = VectorXd::Zero(local_data.rows()).unaryExpr([&](double) { return d(gen); });

        double learning_rate = 0.01;
        double gamma = 0.1;

        // Connect to the server
        boost::asio::io_context io_context;
        tcp::socket socket(io_context);
        socket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 8080));

        // Send vector size to the server
        int vector_size = local_weights.size();
        std::cout << "[DEBUG] Sending vector size: " << vector_size << std::endl;
        boost::asio::write(socket, boost::asio::buffer(&vector_size, sizeof(int)));

        // Train incrementally in batches of 50 samples
        for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
            std::cout << "[INFO] Starting epoch " << epoch + 1 << std::endl;

            bool exit = train_incrementally(local_data, local_labels, local_weights, 
                                            learning_rate, gamma, socket);

            if (exit) {
                std::cout << "[INFO] Exiting after epoch " << epoch + 1 << std::endl;
                break;
            }

            // double loss = compute_loss(local_data, local_labels, local_weights, gamma);
            // std::cout << "[INFO] Epoch " << epoch + 1 << " - Loss: " << loss << std::endl;

            // if (loss < EPSILON) {
            //     std::cout << "[INFO] Converged at epoch " << epoch + 1 << std::endl;
            //     break;
            // }
        }
        socket.close();

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in client: " << e.what() << std::endl;
    }

    return 0;
}
