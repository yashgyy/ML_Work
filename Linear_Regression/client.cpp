
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include "data_loader.cpp"  // Include the data loader

using namespace Eigen;
using boost::asio::ip::tcp;

const int MAX_EPOCHS = 1;
const int TRAIN_BATCH_SIZE = 100;
const int NETWORK_BATCH_SIZE = 100;

// **Mean Squared Error (MSE) Loss Function**
double compute_mse(const MatrixXd& data, const VectorXd& labels, const VectorXd& weights) {
    VectorXd predictions = data * weights;
    VectorXd errors = labels - predictions;
    return (errors.squaredNorm() / labels.size());
}

// **Sends the gradient vector in batches**
void send_in_batches(tcp::socket& socket, const VectorXd& data) {
    int total_size = data.size();
    int sent = 0;

    while (sent < total_size) {
        int batch_size = std::min(NETWORK_BATCH_SIZE, total_size - sent);
        boost::asio::write(socket, boost::asio::buffer(data.data() + sent, batch_size * sizeof(double)));
        sent += batch_size;
    }
}

// **Training and Sending Batches**
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
            VectorXd gradient = VectorXd::Zero(n_features);

            // **Compute the gradient using MSE loss**
            for (int j = 0; j < batch_size; ++j) {
                VectorXd xi = data.row(indices[i + j]);
                double yi = labels(indices[i + j]);
                double prediction = xi.dot(weights);
                double lambda = 0.001; // Regularization strength
                gradient += -2 * xi * (yi - prediction) + lambda * weights;
            }

            gradient /= batch_size;  // Normalize gradient

            // **Send batch size and gradient size before sending updates**
            boost::asio::write(socket, boost::asio::buffer(&batch_size, sizeof(int)));
            boost::asio::write(socket, boost::asio::buffer(&n_features, sizeof(int)));

            // **Send gradient update to the server**
            send_in_batches(socket, gradient);

            // **Receive updated global weights from server**
            boost::asio::read(socket, boost::asio::buffer(weights.data(), weights.size() * sizeof(double)));

            std::cout << "[DEBUG] Updated weights received from server (first 10): "
                      << weights.head(10).transpose() << std::endl;

            // **Compute and display MSE loss after update**
            double mse_loss = compute_mse(data, labels, weights);
            std::cout << "[INFO] Epoch " << epoch + 1 << ", Batch " << (i / TRAIN_BATCH_SIZE) + 1
                      << " - MSE Loss: " << mse_loss << std::endl;
        }
    }
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::socket socket(io_context);

        std::vector<std::vector<double>> features;
        std::vector<double> labels;
        load_data("../Datasets/Regression_1m_v3.csv", features, labels);

        MatrixXd local_data(features.size(), features[0].size());
        VectorXd local_labels(labels.size());

        for (size_t i = 0; i < features.size(); ++i) {
            for (size_t j = 0; j < features[i].size(); ++j) {
                local_data(i, j) = features[i][j];
            }
            local_labels(i) = labels[i];
        }

        // Compute mean and standard deviation per feature
        VectorXd mean = local_data.colwise().mean();
        VectorXd stddev = ((local_data.rowwise() - mean.transpose()).array().square().colwise().mean()).sqrt();

        // Standardize data (zero mean, unit variance)
        for (int i = 0; i < local_data.cols(); ++i) {
            local_data.col(i) = (local_data.col(i).array() - mean(i)) / (stddev(i) + 1e-8);  // Avoid divide-by-zero
        }


        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.01);
        VectorXd weights = VectorXd::Zero(local_data.cols()).unaryExpr([&](double) { return d(gen); });


        socket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 8080));

        train_and_send_batches(socket, local_data, local_labels, weights);

        socket.close();

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in client: " << e.what() << std::endl;
    }

    return 0;
}
