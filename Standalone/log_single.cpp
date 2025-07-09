// Pure Logistic Regression Implementation - No networking, no federated logic
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;

class LogisticRegression {
private:
    VectorXd weights;
    double learning_rate;
    int max_epochs;
    int batch_size;
    
public:
    LogisticRegression(double lr = 0.01, int epochs = 50, int batch = 512)
        : learning_rate(lr), max_epochs(epochs), batch_size(batch) {}

    // Sigmoid function
    double sigmoid(double z) {
        return 1.0 / (1.0 + std::exp(-z));
    }

    // Vectorized sigmoid
    VectorXd sigmoid_vector(const VectorXd& z) {
        VectorXd result(z.size());
        for (int i = 0; i < z.size(); ++i) {
            result(i) = sigmoid(z(i));
        }
        return result;
    }

    void train_batch(const MatrixXd& data, const VectorXd& labels) {
        int n_samples = data.rows();
        int n_features = data.cols();
        
        for (int epoch = 0; epoch < max_epochs; ++epoch) {
            std::cout << "Starting Epoch " << epoch + 1 << std::endl;

            // Shuffle data
            std::vector<int> indices(n_samples);
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);

            for (int i = 0; i < n_samples; i += batch_size) {
                int current_batch_size = std::min(batch_size, n_samples - i);
                VectorXd gradient = VectorXd::Zero(n_features);

                // Compute gradient for this batch
                for (int j = 0; j < current_batch_size; ++j) {
                    VectorXd xi = data.row(indices[i + j]);
                    double yi = labels(indices[i + j]);
                    gradient += xi * (sigmoid(xi.dot(weights)) - yi);
                }

                // Normalize the gradient
                gradient /= current_batch_size;

                // Update weights
                weights -= learning_rate * gradient;
            }
        }
    }

    void fit(const MatrixXd& data, const VectorXd& labels) {
        std::cout << "Training Logistic Regression..." << std::endl;
        
        // Initialize weights to zero
        weights = VectorXd::Zero(data.cols());
        
        // Train using batch processing
        train_batch(data, labels);
        
        std::cout << "Training completed" << std::endl;
    }

    int predict_single(const VectorXd& sample) {
        double dot_product = sample.dot(weights);
        
        if (std::isnan(dot_product) || std::isinf(dot_product)) {
            std::cerr << "[ERROR] Invalid dot product value: " << dot_product << std::endl;
            return -1;  // Return an invalid label for debugging
        }

        double probability = sigmoid(dot_product);
        return (probability >= 0.5) ? 1 : 0;
    }

    VectorXi predict(const MatrixXd& data) {
        VectorXi predictions(data.rows());
        
        for (int i = 0; i < data.rows(); ++i) {
            predictions(i) = predict_single(data.row(i));
        }
        
        return predictions;
    }

    VectorXd predict_proba(const MatrixXd& data) {
        VectorXd probabilities(data.rows());
        
        for (int i = 0; i < data.rows(); ++i) {
            double dot_product = data.row(i).dot(weights);
            probabilities(i) = sigmoid(dot_product);
        }
        
        return probabilities;
    }

    double calculate_accuracy(const MatrixXd& data, const VectorXd& true_labels) {
        VectorXi predictions = predict(data);
        int correct = 0;
        
        for (int i = 0; i < data.rows(); ++i) {
            if (predictions(i) == true_labels(i)) correct++;
        }
        
        return (double)correct / data.rows();
    }

    // Calculate logistic loss (cross-entropy)
    double calculate_loss(const MatrixXd& data, const VectorXd& labels) {
        double loss = 0.0;
        
        for (int i = 0; i < data.rows(); ++i) {
            VectorXd xi = data.row(i);
            double yi = labels(i);
            double probability = sigmoid(xi.dot(weights));
            
            // Add small epsilon to avoid log(0)
            double eps = 1e-15;
            probability = std::max(eps, std::min(1.0 - eps, probability));
            
            loss += -yi * std::log(probability) - (1 - yi) * std::log(1 - probability);
        }
        
        return loss / data.rows();
    }

    void predict_samples(const MatrixXd& test_data) {
        std::cout << "Predicting labels for test data..." << std::endl;

        if (test_data.cols() != weights.size()) {
            std::cerr << "[ERROR] Mismatched dimensions! Test data has " << test_data.cols()
                      << " features, but model expects " << weights.size() << "." << std::endl;
            return;
        }

        for (int i = 0; i < test_data.rows(); ++i) {
            int predicted_label = predict_single(test_data.row(i));
            if (predicted_label == -1) continue;  // Skip invalid predictions
            std::cout << "Sample " << i + 1 << " predicted class: " << predicted_label << std::endl;
        }
    }

    VectorXd get_weights() const { return weights; }
    void set_weights(const VectorXd& new_weights) { weights = new_weights; }
};

// Simple data generator for testing
std::pair<MatrixXd, VectorXd> generate_test_data(int n_samples = 1000, int n_features = 10) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> feature_dist(0.0, 1.0);
    
    MatrixXd data(n_samples, n_features);
    VectorXd labels(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            data(i, j) = feature_dist(gen);
        }
        // Simple rule: if sum of first 3 features > 0, label = 1, else 0
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
    
    // Create and train Logistic Regression
    LogisticRegression lr(0.01, 50, 512);
    lr.fit(train_data, train_labels);
    
    // Test accuracy
    double train_accuracy = lr.calculate_accuracy(train_data, train_labels);
    double test_accuracy = lr.calculate_accuracy(test_data, test_labels);
    
    std::cout << "Training accuracy: " << train_accuracy * 100 << "%" << std::endl;
    std::cout << "Test accuracy: " << test_accuracy * 100 << "%" << std::endl;
    
    // Calculate loss
    double train_loss = lr.calculate_loss(train_data, train_labels);
    double test_loss = lr.calculate_loss(test_data, test_labels);
    
    std::cout << "Training loss: " << train_loss << std::endl;
    std::cout << "Test loss: " << test_loss << std::endl;
    
    // Test single prediction
    VectorXd sample = test_data.row(0);
    int prediction = lr.predict_single(sample);
    std::cout << "Single prediction for first test sample: " << prediction 
              << " (actual: " << test_labels(0) << ")" << std::endl;
    
    // Show prediction probabilities for first few samples
    VectorXd probabilities = lr.predict_proba(test_data.topRows(5));
    std::cout << "\nPrediction probabilities for first 5 test samples:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Sample " << i + 1 << ": " << probabilities(i) 
                  << " (actual: " << test_labels(i) << ")" << std::endl;
    }
    
    return 0;
}