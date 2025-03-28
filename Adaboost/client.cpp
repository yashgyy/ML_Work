// Updated Federated AdaBoost Client with Epoch-based Training and Global Redistribution
#include <iostream>
#include <vector>
#include <random>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include "data_loader.cpp"

using namespace Eigen;
using boost::asio::ip::tcp;

const int NUM_EPOCHS = 5;
const int LEARNERS_PER_EPOCH = 2;

struct WeakLearner {
    int feature_index;
    double threshold;
    double alpha;
};

WeakLearner train_weak_learner(const MatrixXd& data, const VectorXd& labels, const VectorXd& weights) {
    int n_samples = data.rows();
    int n_features = data.cols();
    WeakLearner best_learner = {0, 0, 0};
    double best_error = std::numeric_limits<double>::max();

    for (int feature_index = 0; feature_index < n_features; ++feature_index) {
        for (int i = 0; i < n_samples; ++i) {
            double threshold = data(i, feature_index);
            VectorXd predictions = (data.col(feature_index).array() <= threshold).cast<double>() * 2 - 1;
            double weighted_error = (weights.array() * (predictions.array() != labels.array()).cast<double>()).sum();

            if (weighted_error < best_error) {
                best_error = weighted_error;
                best_learner = {feature_index, threshold, 0};
            }
        }
    }
    best_learner.alpha = 0.5 * std::log((1 - best_error) / (best_error + 1e-10));
    return best_learner;
}

std::vector<WeakLearner> train_adaboost(const MatrixXd& data, const VectorXd& labels, int num_learners) {
    std::vector<WeakLearner> learners;
    VectorXd weights = VectorXd::Ones(data.rows()) / data.rows();

    for (int t = 0; t < num_learners; ++t) {
        WeakLearner learner = train_weak_learner(data, labels, weights);
        learners.push_back(learner);

        VectorXd predictions = (data.col(learner.feature_index).array() <= learner.threshold).cast<double>() * 2 - 1;
        weights.array() *= (-learner.alpha * labels.array() * predictions.array()).exp();
        weights /= weights.sum();
    }
    return learners;
}

std::vector<double> serialize_learners(const std::vector<WeakLearner>& learners) {
    std::vector<double> serialized;
    for (const auto& learner : learners) {
        serialized.push_back(learner.feature_index);
        serialized.push_back(learner.threshold);
        serialized.push_back(learner.alpha);
    }
    return serialized;
}

std::vector<WeakLearner> deserialize_learners(const std::vector<double>& serialized) {
    std::vector<WeakLearner> learners;
    for (size_t i = 0; i + 2 < serialized.size(); i += 3) {
        WeakLearner learner;
        learner.feature_index = static_cast<int>(serialized[i]);
        learner.threshold = serialized[i + 1];
        learner.alpha = serialized[i + 2];
        learners.push_back(learner);
    }
    return learners;
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::socket socket(io_context);
        socket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 8080));

        std::vector<std::vector<float>> features;
        std::vector<int> labels;
        load_data("../Datasets/santander-customer-transaction-prediction.csv", features, labels);
        std::cout<<"Data Read Succesfully"<<std::endl;
        MatrixXd data(features.size(), features[0].size());
        VectorXd label_vec(labels.size());
        for (size_t i = 0; i < features.size(); ++i) {
            for (size_t j = 0; j < features[i].size(); ++j) {
                data(i, j) = features[i][j];
            }
            label_vec(i) = labels[i];
        }

        for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
            std::cout << "[INFO] Epoch " << epoch + 1 << " begins...\n";
            std::vector<WeakLearner> learners = train_adaboost(data, label_vec, LEARNERS_PER_EPOCH);
            std::vector<double> serialized = serialize_learners(learners);

            int num_learners = learners.size();
            int vec_size = serialized.size();

            boost::asio::write(socket, boost::asio::buffer(&num_learners, sizeof(int)));
            boost::asio::write(socket, boost::asio::buffer(&vec_size, sizeof(int)));
            boost::asio::write(socket, boost::asio::buffer(serialized.data(), vec_size * sizeof(double)));

            std::cout << "[INFO] Sent " << num_learners << " learners to server in epoch " << epoch + 1 << ".\n";

            // Receive global model from server
            int global_vec_size = 0;
            boost::asio::read(socket, boost::asio::buffer(&global_vec_size, sizeof(int)));
            std::vector<double> global_serialized(global_vec_size);
            boost::asio::read(socket, boost::asio::buffer(global_serialized.data(), global_vec_size * sizeof(double)));

            std::vector<WeakLearner> global_model = deserialize_learners(global_serialized);
            std::cout << "[INFO] Received global model with " << global_model.size() << " weak learners.\n";
        }

        socket.close();

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in client: " << e.what() << std::endl;
    }
    return 0;
}
