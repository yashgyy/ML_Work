// Refactored Federated Linear SVM Client (based on K_client structure)
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <boost/asio.hpp>
#include <Eigen/Dense>
//#include "data_loader.cpp"
//#include "data_loader_susy.cpp"
#include "data_loader_higgs.cpp"

using namespace Eigen;
using boost::asio::ip::tcp;

const double LEARNING_RATE = 0.01;
const int MAX_EPOCHS = 1;
const int TRAIN_BATCH_SIZE = 2;
const int NETWORK_BATCH_SIZE = 100;

// Hinge loss derivative for Linear SVM
VectorXd compute_svm_gradient(const MatrixXd& X, const VectorXd& y, const VectorXd& weights) {
    VectorXd gradient = VectorXd::Zero(weights.size());
    int n = X.rows();

    for (int i = 0; i < n; ++i) {
        VectorXd xi = X.row(i);
        double yi = y(i);
        if (yi * xi.dot(weights) < 1) {
            gradient -= yi * xi;
        }
    }

    gradient /= n; // Average over batch
    return gradient;
}

void send_in_batches(tcp::socket& socket, const VectorXd& data) {
    int total_size = data.size();
    int sent = 0;
    while (sent < total_size) {
        int batch_size = std::min(NETWORK_BATCH_SIZE, total_size - sent);
        boost::asio::write(socket, boost::asio::buffer(data.data() + sent, batch_size * sizeof(double)));
        sent += batch_size;
    }
}

void train_and_send_batches(tcp::socket& socket, MatrixXd& data, VectorXd& labels, VectorXd& weights) {
    int n_samples = data.rows();
    int n_features = data.cols();

    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
        std::cout << "[INFO] Starting Epoch " << epoch + 1 << std::endl;

        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        for (int i = 0; i < n_samples; i += TRAIN_BATCH_SIZE) {
            int batch_size = std::min(TRAIN_BATCH_SIZE, n_samples - i);
            MatrixXd batch_X(batch_size, n_features);
            VectorXd batch_y(batch_size);

            for (int j = 0; j < batch_size; ++j) {
                batch_X.row(j) = data.row(indices[i + j]);
                batch_y(j) = labels(indices[i + j]);
            }

            VectorXd gradient = compute_svm_gradient(batch_X, batch_y, weights);

            // Send batch metadata
            boost::asio::write(socket, boost::asio::buffer(&batch_size, sizeof(int)));
            boost::asio::write(socket, boost::asio::buffer(&n_features, sizeof(int)));

            // Send gradient
            send_in_batches(socket, gradient);

            // Receive updated global model
            boost::asio::read(socket, boost::asio::buffer(weights.data(), weights.size() * sizeof(double)));

            std::cout << "[DEBUG] Updated weights received: " << weights.transpose() << std::endl;
        }
    }
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::socket socket(io_context);

        std::vector<std::vector<float>> features;
        std::vector<int> labels;
        //load_data("../Datasets/santander-customer-transaction-prediction.csv", features, labels);
        //load_data("../Datasets/SUSY.csv", features, labels);
        load_data("../Datasets/HIGGS.csv", features, labels);


        MatrixXd data(features.size(), features[0].size());
        VectorXd label_vec(labels.size());
        for (size_t i = 0; i < features.size(); ++i) {
            for (size_t j = 0; j < features[i].size(); ++j) {
                data(i, j) = features[i][j];
            }
            label_vec(i) = labels[i];
        }

        VectorXd weights = VectorXd::Random(data.cols());
        
  

        socket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 8080));

        train_and_send_batches(socket, data, label_vec, weights);
        socket.close();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in client: " << e.what() << std::endl;
    }
    return 0;
}
