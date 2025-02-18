#include <iostream>
#include <vector>
#include <random>
#include <boost/asio.hpp>
#include <Eigen/Dense>

using namespace Eigen;
using boost::asio::ip::tcp;

// Structure to represent a weak learner (decision stump)
struct WeakLearner {
    int feature_index;
    double threshold;
    double alpha; // Weight of the weak learner in the ensemble
};

// Train a weak learner (decision stump)
WeakLearner train_weak_learner(const MatrixXd& data, const VectorXd& labels, const VectorXd& weights) {
    int n_samples = data.rows();
    int n_features = data.cols();
    WeakLearner best_learner = {0, 0, 0};
    double best_error = std::numeric_limits<double>::max();

    // Loop over each feature to find the best threshold
    for (int feature_index = 0; feature_index < n_features; ++feature_index) {
        for (int i = 0; i < n_samples; ++i) {
            double threshold = data(i, feature_index);
            VectorXd predictions = (data.col(feature_index).array() <= threshold).cast<double>() * 2 - 1;

            // Compute weighted error
            double weighted_error = (weights.array() * (predictions.array() != labels.array()).cast<double>()).sum();

            if (weighted_error < best_error) {
                best_error = weighted_error;
                best_learner = {feature_index, threshold, 0};
            }
        }
    }

    // Calculate the alpha (weight of the weak learner)
    best_learner.alpha = 0.5 * std::log((1 - best_error) / (best_error + 1e-10));
    return best_learner;
}

// Train AdaBoost on local data
std::vector<WeakLearner> train_adaboost(const MatrixXd& data, const VectorXd& labels, int num_learners) {
    std::vector<WeakLearner> learners;
    VectorXd weights = VectorXd::Ones(data.rows()) / data.rows(); // Initialize weights

    for (int t = 0; t < num_learners; ++t) {
        WeakLearner learner = train_weak_learner(data, labels, weights);
        learners.push_back(learner);

        // Update sample weights
        VectorXd predictions = (data.col(learner.feature_index).array() <= learner.threshold).cast<double>() * 2 - 1;
        weights.array() *= (-learner.alpha * labels.array() * predictions.array()).exp();
        weights /= weights.sum();  // Normalize weights
    }

    return learners;
}

// Serialize weak learners
std::vector<double> serialize_learners(const std::vector<WeakLearner>& learners) {
    std::vector<double> serialized;
    for (const auto& learner : learners) {
        serialized.push_back(learner.feature_index);
        serialized.push_back(learner.threshold);
        serialized.push_back(learner.alpha);
    }
    return serialized;
}

int main() {
    boost::asio::io_service io_service;
    tcp::socket socket(io_service);
    tcp::resolver resolver(io_service);
    boost::asio::connect(socket, resolver.resolve({"127.0.0.1", "8080"}));

    // Load local data
    MatrixXd data(4, 2); // Example data (4 samples, 2 features)
    data << 1, 2,
            2, 1,
            3, 4,
            4, 3;
    VectorXd labels(4);
    labels << 1, 1, -1, -1;

    // Train AdaBoost on local data
    std::vector<WeakLearner> learners = train_adaboost(data, labels, 5);

    // Serialize the learners
    std::vector<double> serialized_learners = serialize_learners(learners);

    // Send the number of learners to the server
    int num_learners = learners.size();
    boost::asio::write(socket, boost::asio::buffer(&num_learners, sizeof(int)));

    // Send the serialized learners to the server
    boost::asio::write(socket, boost::asio::buffer(serialized_learners.data(), serialized_learners.size() * sizeof(double)));

    std::cout << "AdaBoost model has been sent to the server.\n";

    return 0;
}
