#include <iostream>
#include <vector>
#include <thread>
#include <boost/asio.hpp>
#include <mutex>
#include <Eigen/Dense>
#include <map>

using namespace Eigen;
using boost::asio::ip::tcp;

// Structure to represent a weak learner (decision stump)
struct WeakLearner {
    int feature_index;
    double threshold;
    double alpha; // Weight of the weak learner in the ensemble
};

std::mutex model_mutex;
std::vector<WeakLearner> aggregated_learners; // Store all weak learners from clients

// Deserialize weak learners
std::vector<WeakLearner> deserialize_learners(const std::vector<double>& serialized_learners, int num_learners) {
    std::vector<WeakLearner> learners;
    for (int i = 0; i < num_learners; ++i) {
        WeakLearner learner;
        learner.feature_index = serialized_learners[3 * i];
        learner.threshold = serialized_learners[3 * i + 1];
        learner.alpha = serialized_learners[3 * i + 2];
        learners.push_back(learner);
    }
    return learners;
}

// Predict using AdaBoost ensemble
double predict_adaboost(const MatrixXd& data, const std::vector<WeakLearner>& learners, int sample_index) {
    double prediction = 0.0;
    for (const auto& learner : learners) {
        double stump_prediction = (data(sample_index, learner.feature_index) <= learner.threshold) ? 1 : -1;
        prediction += learner.alpha * stump_prediction;
    }
    return prediction >= 0 ? 1 : -1;
}

// Handle client function: Deserialize and store weak learners from the client
void handle_client(tcp::socket socket) {
    try {
        std::cout << "[DEBUG] Handling new client connection.\n";

        // Read the number of learners from the client
        int num_learners;
        boost::asio::read(socket, boost::asio::buffer(&num_learners, sizeof(int)));

        // Read the serialized learners
        std::vector<double> serialized_learners(num_learners * 3); // Each learner has 3 components (feature_index, threshold, alpha)
        boost::asio::read(socket, boost::asio::buffer(serialized_learners.data(), serialized_learners.size() * sizeof(double)));

        // Deserialize the learners
        std::vector<WeakLearner> learners = deserialize_learners(serialized_learners, num_learners);

        // Lock and aggregate the weak learners
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            aggregated_learners.insert(aggregated_learners.end(), learners.begin(), learners.end());
        }

        std::cout << "[DEBUG] Received and stored weak learners from a client.\n";

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}

// Function to perform predictions
void perform_predictions() {
    if (aggregated_learners.size() >= 5) {  // Example condition to start predictions
        MatrixXd test_data(4, 2);  // Example test data
        test_data << 1, 2,
                     2, 1,
                     3, 4,
                     4, 3;

        std::cout << "Predictions for test data:\n";
        for (int i = 0; i < test_data.rows(); ++i) {
            double prediction = predict_adaboost(test_data, aggregated_learners, i);
            std::cout << "Sample " << i << ": " << prediction << std::endl;
        }
    }
}

int main() {
    boost::asio::io_service io_service;
    tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), 8080));

    std::cout << "Server is running on port 8080...\n";

    while (true) {
        tcp::socket socket(io_service);
        acceptor.accept(socket);

        // Handle each client in a separate thread
        std::thread(handle_client, std::move(socket)).detach();

        // After handling clients, call the prediction function if conditions are met
        perform_predictions();
    }

    return 0;
}
