// Pure Linear SVM Implementation - No networking, no federated logic
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <Eigen/Dense>

using namespace Eigen;

class LinearSVM {
private:
    VectorXd weights;
    double learning_rate;
    int max_epochs;
    int batch_size;
    
public:
    LinearSVM(double lr = 0.01, int epochs = 50, int batch = 512)
        : learning_rate(lr), max_epochs(epochs), batch_size(batch) {}

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

            // Process in batches
            for (int i = 0; i < n_samples; i += batch_size) {
                int current_batch_size = std::min(batch_size, n_samples - i);
                MatrixXd batch_X(current_batch_size, n_features);
                VectorXd batch_y(current_batch_size);

                // Create batch
                for (int j = 0; j < current_batch_size; ++j) {
                    batch_X.row(j) = data.row(indices[i + j]);
                    batch_y(j) = labels(indices[i + j]);
                }

                // Compute gradient and update weights
                VectorXd gradient = compute_svm_gradient(batch_X, batch_y, weights);
                weights -= learning_rate * gradient;
            }
        }
    }

    void fit(const MatrixXd& data, const VectorXd& labels) {
        std::cout << "Training Linear SVM..." << std::endl;
        
        // Initialize weights randomly
        weights = VectorXd::Random(data.cols());
        
        // Train using batch processing
        train_batch(data, labels);
        
        std::cout << "Training completed" << std::endl;
    }

    int predict_single(const VectorXd& sample) {
        return (sample.dot(weights) > 0) ? 1 : -1;
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

    // Calculate hinge loss
    double calculate_hinge_loss(const MatrixXd& data, const VectorXd& labels) {
        double loss = 0.0;
        
        for (int i = 0; i < data.rows(); ++i) {
            VectorXd xi = data.row(i);
            double yi = labels(i);
            double margin = yi * xi.dot(weights);
            if (margin < 1) {
                loss += (1 - margin);
            }
        }
        
        return loss / data.rows();
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
        // Simple rule: if sum of first 3 features > 0, label = 1, else -1
        labels(i) = (data(i, 0) + data(i, 1) + data(i, 2) > 0) ? 1 : -1;
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
    
    // Create and train Linear SVM
    LinearSVM svm(0.01, 50, 512);
    svm.fit(train_data, train_labels);
    
    // Test accuracy
    double train_accuracy = svm.calculate_accuracy(train_data, train_labels);
    double test_accuracy = svm.calculate_accuracy(test_data, test_labels);
    
    std::cout << "Training accuracy: " << train_accuracy * 100 << "%" << std::endl;
    std::cout << "Test accuracy: " << test_accuracy * 100 << "%" << std::endl;
    
    // Calculate loss
    double train_loss = svm.calculate_hinge_loss(train_data, train_labels);
    double test_loss = svm.calculate_hinge_loss(test_data, test_labels);
    
    std::cout << "Training hinge loss: " << train_loss << std::endl;
    std::cout << "Test hinge loss: " << test_loss << std::endl;
    
    // Test single prediction
    VectorXd sample = test_data.row(0);
    int prediction = svm.predict_single(sample);
    std::cout << "Single prediction for first test sample: " << prediction 
              << " (actual: " << test_labels(0) << ")" << std::endl;
    
    return 0;
}