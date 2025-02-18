#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <boost/asio.hpp>
#include <Eigen/Dense>

using namespace Eigen;
using boost::asio::ip::tcp;

// Shuffle data to simulate different local datasets for each client
MatrixXd load_local_data() {
    MatrixXd data(4, 2); // 4 samples, 2 features
    data << 1, 2,
            2, 1,
            3, 4,
            4, 3;

    // Shuffle rows of data to simulate different client data
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<int> indices = {0, 1, 2, 3};
    std::shuffle(indices.begin(), indices.end(), g);

    MatrixXd shuffled_data(4, 2);
    for (int i = 0; i < 4; ++i) {
        shuffled_data.row(i) = data.row(indices[i]);
    }

    return shuffled_data;
}

VectorXd load_local_labels() {
    VectorXd labels(4); // 4 samples
    labels << 1, 1, -1, -1;
    return labels;
}

// Perform one step of training using SGD
VectorXd train_local_svm(const MatrixXd& data, const VectorXd& labels, VectorXd& weights, double learning_rate) {
    int n_samples = data.rows();
    int n_features = data.cols();

    VectorXd gradient = VectorXd::Zero(n_features);

    for (int i = 0; i < n_samples; ++i) {
        VectorXd xi = data.row(i);
        double yi = labels(i);

        // Calculate gradient for hinge loss
        if (yi * (xi.dot(weights)) < 1) {
            gradient += -yi * xi;
        }
    }

    // Update weights
    weights -= learning_rate * gradient;

    return weights;
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::socket socket(io_context);
        socket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 8080));

        // Load local data
        MatrixXd local_data = load_local_data();
        VectorXd local_labels = load_local_labels();

        // Initialize local model weights with small random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.01); // Small random values
        VectorXd local_weights = VectorXd::Zero(local_data.cols()).unaryExpr([&](double dummy) { return d(gen); });

        double learning_rate = 0.01;

        // Perform local training (single step for simplicity)
        VectorXd updated_weights = train_local_svm(local_data, local_labels, local_weights, learning_rate);

        // Print local update
        std::cout << "Local model update: " << updated_weights.transpose() << std::endl;

        // Send the size of the vector first
        int vector_size = updated_weights.size();
        boost::asio::write(socket, boost::asio::buffer(&vector_size, sizeof(int)));

        // Send local update (weights) to server
        boost::asio::write(socket, boost::asio::buffer(updated_weights.data(), updated_weights.size() * sizeof(double)));

        // Receive updated global model from server
        VectorXd global_model(local_weights.size());
        boost::asio::read(socket, boost::asio::buffer(global_model.data(), global_model.size() * sizeof(double)));

        std::cout << "Received updated global model: " << global_model.transpose() << std::endl;
    } catch (std::exception& e) {
        std::cerr << "Exception in client: " << e.what() << std::endl;
    }

    return 0;
}
