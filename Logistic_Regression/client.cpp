#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <boost/asio.hpp>
#include <Eigen/Dense>
//#include "data_loader.cpp" // Include the data loader
//#include "data_loader_susy.cpp"
#include "data_loader_higgs.cpp"
// g++ client_updated.cpp -o client  -I /usr/include/eigen3

using namespace Eigen;
using boost::asio::ip::tcp;

const double LEARNING_RATE = 0.01;
const int MAX_EPOCHS = 1;
const int TRAIN_BATCH_SIZE = 100;
const int NETWORK_BATCH_SIZE = 100;

// Sigmoid function
double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

int predict(const VectorXd& sample, const VectorXd& weights) {
    double dot_product = sample.dot(weights);
    
    if (std::isnan(dot_product) || std::isinf(dot_product)) {
        std::cerr << "[ERROR] Invalid dot product value: " << dot_product << std::endl;
        return -1;  // Return an invalid label for debugging
    }

    double probability = sigmoid(dot_product);
    return (probability >= 0.5) ? 1 : 0;
}

void predict_samples(const MatrixXd& test_data, const VectorXd& weights) {
    std::cout << "[INFO] Predicting labels for test data..." << std::endl;

    if (test_data.cols() != weights.size()) {
        std::cerr << "[ERROR] Mismatched dimensions! Test data has " << test_data.cols()
                  << " features, but model expects " << weights.size() << "." << std::endl;
        return;
    }

    for (int i = 0; i < test_data.rows(); ++i) {
        int predicted_label = predict(test_data.row(i), weights);
        if (predicted_label == -1) continue;  // Skip invalid predictions
        std::cout << "Sample " << i + 1 << " predicted class: " << predicted_label << std::endl;
    }
}


// Sends the weight vector in batches over the network
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

        // Shuffle data before each epoch
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        for (int i = 0; i < n_samples; i += TRAIN_BATCH_SIZE) {
            int batch_size = std::min(TRAIN_BATCH_SIZE, n_samples - i);
            VectorXd gradient = VectorXd::Zero(n_features);

            for (int j = 0; j < batch_size; ++j) {
                VectorXd xi = data.row(indices[i + j]);
                double yi = labels(indices[i + j]);
                gradient += xi * (sigmoid(xi.dot(weights)) - yi);
            }

            // Normalize the gradient
            gradient /= batch_size;

            // Send batch metadata
            boost::asio::write(socket, boost::asio::buffer(&batch_size, sizeof(int)));
            boost::asio::write(socket, boost::asio::buffer(&n_features, sizeof(int)));

            // Send weight updates
            send_in_batches(socket, gradient);

            // Receive updated global model
            boost::asio::read(socket, boost::asio::buffer(weights.data(), weights.size() * sizeof(double)));

            std::cout << "[DEBUG] Updated weights received from server (first 10): " 
                      << weights.head(10).transpose() << std::endl;
        }
    }
        std::vector<std::vector<float>> test_features;
        std::vector<int> test_labels;
        //load_data("../Datasets/santander-customer-transaction-prediction.csv", test_features, test_labels);
        //load_data("../Datasets/SUSY.csv", test_features, test_labels);
        load_data("../Datasets/HIGGS.csv", test_features, test_labels);

        MatrixXd test_data(test_features.size(), test_features[0].size());
        for (size_t i = 0; i < test_features.size(); ++i) {
            for (size_t j = 0; j < test_features[i].size(); ++j) {
                test_data(i, j) = test_features[i][j];
            }
        }

        predict_samples(test_data, weights);
    
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

        // **Convert std::vector<std::vector<float>> to Eigen::MatrixXd**
        MatrixXd local_data(features.size(), features[0].size());
        VectorXd local_labels(labels.size());
        
        for (size_t i = 0; i < features.size(); ++i) {
            for (size_t j = 0; j < features[i].size(); ++j) {
                local_data(i, j) = features[i][j];  // Convert to Eigen::MatrixXd
            }
            local_labels(i) = labels[i];  // Convert to Eigen::VectorXd
        }

        // **Declare and Initialize weights**
        VectorXd weights = VectorXd::Zero(local_data.cols());

        // **Connect to server**
        socket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 8080));

        // **Start training and sending updates**
        train_and_send_batches(socket, local_data, local_labels, weights);
        

        socket.close();

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in client: " << e.what() << std::endl;
    }

    return 0;
}
