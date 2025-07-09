// Pure Random Forest Implementation - No networking, no federated logic
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <Eigen/Dense>

using namespace Eigen;

struct DecisionTree {
    int feature_index;
    float threshold;
    int class_label;
};

class RandomForest {
private:
    int num_trees;
    int num_epochs;
    std::vector<DecisionTree> forest;
    
public:
    RandomForest(int trees = 100, int epochs = 50) 
        : num_trees(trees), num_epochs(epochs) {}

    std::vector<DecisionTree> train_trees(const MatrixXd& data, const VectorXd& labels, int trees_to_train) {
        std::vector<DecisionTree> local_forest;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, data.rows() - 1);

        for (int t = 0; t < trees_to_train; ++t) {
            // Bootstrap sampling
            std::vector<int> samples;
            for (int i = 0; i < data.rows() / 2; ++i)
                samples.push_back(dis(gen));

            // Find best split
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

            // Determine majority class
            int majority_class = 1;
            int count_1 = 0;
            for (int i : samples) if (labels(i) == 1) count_1++;
            majority_class = (count_1 > samples.size() / 2) ? 1 : 0;

            local_forest.push_back({best_feature, best_threshold, majority_class});
        }
        return local_forest;
    }

    void fit(const MatrixXd& data, const VectorXd& labels) {
        std::cout << "Training Random Forest with " << num_trees << " trees..." << std::endl;
        
        forest.clear();
        int trees_per_epoch = num_trees / num_epochs;
        if (trees_per_epoch == 0) trees_per_epoch = 1;
        
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::vector<DecisionTree> epoch_trees = train_trees(data, labels, trees_per_epoch);
            forest.insert(forest.end(), epoch_trees.begin(), epoch_trees.end());
            
            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch + 1 << " completed. Forest size: " << forest.size() << std::endl;
            }
        }
        
        std::cout << "Training completed. Final forest size: " << forest.size() << std::endl;
    }

    int predict_single(const VectorXd& sample) {
        if (forest.empty()) return 0;
        
        int votes_0 = 0, votes_1 = 0;
        
        for (const auto& tree : forest) {
            if (sample(tree.feature_index) <= tree.threshold) {
                if (tree.class_label == 0) votes_0++;
                else votes_1++;
            } else {
                if (tree.class_label == 0) votes_0++;
                else votes_1++;
            }
        }
        
        return (votes_1 > votes_0) ? 1 : 0;
    }

    VectorXi predict(const MatrixXd& data) {
        VectorXi predictions(data.rows());
        
        for (int i = 0; i < data.rows(); ++i) {
            predictions(i) = predict_single(data.row(i));
        }
        
        return predictions;
    }

    double calculate_accuracy(const MatrixXd& data, const VectorXd& true_labels) {
        VectorXi predictions = predict(data);
        int correct = 0;
        
        for (int i = 0; i < data.rows(); ++i) {
            if (predictions(i) == true_labels(i)) correct++;
        }
        
        return (double)correct / data.rows();
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
            trees.push_back({static_cast<int>(serialized[i]), 
                           static_cast<float>(serialized[i + 1]), 
                           static_cast<int>(serialized[i + 2])});
        }
        return trees;
    }

    int get_forest_size() const { return forest.size(); }
};

// Simple data generator for testing
std::pair<MatrixXd, VectorXd> generate_test_data(int n_samples = 1000, int n_features = 10) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> feature_dist(0.0, 1.0);
    std::uniform_int_distribution<> label_dist(0, 1);
    
    MatrixXd data(n_samples, n_features);
    VectorXd labels(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            data(i, j) = feature_dist(gen);
        }
        // Simple rule: if sum of first 3 features > 0, label = 1
        labels(i) = (data(i, 0) + data(i, 1) + data(i, 2) > 0) ? 1 : 0;
    }
    
    return {data, labels};
}

int main() {
    // Generate test data
    auto [data, labels] = generate_test_data(1000, 10);
    
    // Split into train and test
    int train_size = 800;
    MatrixXd train_data = data.topRows(train_size);
    VectorXd train_labels = labels.head(train_size);
    MatrixXd test_data = data.bottomRows(data.rows() - train_size);
    VectorXd test_labels = labels.tail(labels.size() - train_size);
    
    // Create and train Random Forest
    RandomForest rf(100, 20);  // 100 trees, 20 epochs
    rf.fit(train_data, train_labels);
    
    // Test accuracy
    double train_accuracy = rf.calculate_accuracy(train_data, train_labels);
    double test_accuracy = rf.calculate_accuracy(test_data, test_labels);
    
    std::cout << "Training accuracy: " << train_accuracy * 100 << "%" << std::endl;
    std::cout << "Test accuracy: " << test_accuracy * 100 << "%" << std::endl;
    
    // Test single prediction
    VectorXd sample = test_data.row(0);
    int prediction = rf.predict_single(sample);
    std::cout << "Single prediction for first test sample: " << prediction 
              << " (actual: " << test_labels(0) << ")" << std::endl;
    
    return 0;
}