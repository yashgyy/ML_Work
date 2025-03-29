// Federated Random Forest Client (Epoch-based, Batch-wise)
#include <iostream>
#include <vector>
#include <random>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include "data_loader.cpp"
// g++ client_updated.cpp -o client  -I /usr/include/eigen3

using namespace Eigen;
using boost::asio::ip::tcp;

const int NUM_EPOCHS = 5;
const int TREES_PER_EPOCH = 2;

struct DecisionTree {
    int feature_index;
    float threshold;
    int class_label; // simple leaf label
};

std::vector<DecisionTree> train_trees(const MatrixXd& data, const VectorXd& labels, int num_trees) {
    std::vector<DecisionTree> forest;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.rows() - 1);

    for (int t = 0; t < num_trees; ++t) {
        std::vector<int> samples;
        for (int i = 0; i < data.rows() / 2; ++i)
            samples.push_back(dis(gen));

        int best_feature = 0;
        float best_threshold = 0;
        double best_score = 1.0;

        for (int f = 0; f < data.cols(); ++f) {
            float threshold = data(samples[0], f);
            int left_1 = 0, left_total = 0;
            for (int i : samples) {
                if (data(i, f) <= threshold) {
                    left_1 += (labels(i) == 1);
                    left_total++;
                }
            }
            if (left_total == 0 || left_total == samples.size()) continue;
            double p = (double)left_1 / left_total;
            double gini = 1.0 - (p * p + (1 - p) * (1 - p));
            if (gini < best_score) {
                best_score = gini;
                best_feature = f;
                best_threshold = threshold;
            }
        }

        int majority_class = 1;
        int count_1 = 0;
        for (int i : samples) if (labels(i) == 1) count_1++;
        majority_class = (count_1 > samples.size() / 2) ? 1 : 0;

        forest.push_back({best_feature, best_threshold, majority_class});
    }
    return forest;
}

std::vector<double> serialize_trees(const std::vector<DecisionTree>& trees) {
    std::vector<double> serialized;
    for (const auto& tree : trees) {
        serialized.push_back(tree.feature_index);
        serialized.push_back(tree.threshold);
        serialized.push_back(tree.class_label);
    }
    return serialized;
}

std::vector<DecisionTree> deserialize_trees(const std::vector<double>& serialized) {
    std::vector<DecisionTree> trees;
    for (size_t i = 0; i + 2 < serialized.size(); i += 3) {
        trees.push_back({static_cast<int>(serialized[i]), static_cast<float>(serialized[i + 1]), static_cast<int>(serialized[i + 2])});
    }
    return trees;
}

int main() {
    try {
        std::vector<std::vector<float>> features;
        std::vector<int> labels;
        load_data("../Datasets/santander-customer-transaction-prediction.csv", features, labels);

        MatrixXd data(features.size(), features[0].size());
        VectorXd label_vec(labels.size());
        for (size_t i = 0; i < features.size(); ++i) {
            for (size_t j = 0; j < features[i].size(); ++j)
                data(i, j) = features[i][j];
            label_vec(i) = labels[i];
        }

        boost::asio::io_context io_context;
        tcp::socket socket(io_context);
        socket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 8080));

        for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
            std::vector<DecisionTree> trees = train_trees(data, label_vec, TREES_PER_EPOCH);
            std::vector<double> serialized = serialize_trees(trees);

            int num_trees = trees.size();
            int vec_size = serialized.size();

            boost::asio::write(socket, boost::asio::buffer(&num_trees, sizeof(int)));
            boost::asio::write(socket, boost::asio::buffer(&vec_size, sizeof(int)));
            boost::asio::write(socket, boost::asio::buffer(serialized.data(), vec_size * sizeof(double)));

            int global_vec_size;
            boost::asio::read(socket, boost::asio::buffer(&global_vec_size, sizeof(int)));
            std::vector<double> global_serialized(global_vec_size);
            boost::asio::read(socket, boost::asio::buffer(global_serialized.data(), global_vec_size * sizeof(double)));

            std::vector<DecisionTree> global_forest = deserialize_trees(global_serialized);
            std::cout << "[INFO] Epoch " << epoch + 1 << ": Global forest size = " << global_forest.size() << std::endl;
        }

        socket.close();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in client: " << e.what() << std::endl;
    }
    return 0;
}
