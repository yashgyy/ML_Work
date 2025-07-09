// Pure Kernel SVM Implementation - No networking, no federated logic
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <Eigen/Dense>

using namespace Eigen;

class KernelSVM {
private:
    VectorXd weights;
    double learning_rate;
    double gamma;
    int max_epochs;
    int batch_size;
    double epsilon;
    
public:
    KernelSVM(double lr = 0.01, double g = 0.1, int epochs = 50, int batch = 30, double eps = 1e-5)
        : learning_rate(lr), gamma(g), max_epochs(epochs), batch_size(batch), epsilon(eps) {}

    double rbf_kernel(const VectorXd& x1, const VectorXd& x2, double gamma_val) {
        return std::exp(-gamma_val * (x1 - x2).squaredNorm());
    }

    bool train_incrementally(const MatrixXd& data, const VectorXd& labels) {
        int n_samples = data.rows();
        static int last_sample = 0;
        VectorXd gradient = VectorXd::Zero(weights.size());
        int current_sample = last_sample;
        int processed_samples = 0;

        while (processed_samples < n_samples) {
            if (current_sample >= n_samples) return true; // All samples processed

            VectorXd xi = data.row(current_sample);
            double yi = labels(current_sample);
            double kernel_output = 0.0;

            // Calculate kernel output
            for (int j = 0; j < weights.size(); ++j)
                kernel_output += rbf_kernel(xi, data.row(j), gamma) * weights(j);

            // Update gradient if margin violation
            if (yi * kernel_output < 1) {
                for (int j = 0; j < weights.size(); ++j)
                    gradient(j) += -yi * rbf_kernel(data.row(j), xi, gamma);
            }

            // Update weights
            weights -= learning_rate * gradient;
            current_sample++;
            processed_samples++;

            // Reset gradient after batch
            if (processed_samples % batch_size == 0) {
                gradient.setZero();
            }
        }

        last_sample = current_sample;
        return false;
    }

    void fit(const MatrixXd& data, const VectorXd& labels) {
        std::cout << "Training Kernel SVM..." << std::endl;
        
        // Initialize weights randomly
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.01);
        weights = VectorXd::Zero(data.rows()).unaryExpr([&](double) { return d(gen); });

        for (int epoch = 0; epoch < max_epochs; ++epoch) {
            bool exit = train_incrementally(data, labels);
            if (exit) break;
            
            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch + 1 << " completed" << std::endl;
            }
        }
        
        std::cout << "Training completed" << std::endl;
    }

    int predict_single(const VectorXd& sample, const MatrixXd& training_data) {
        double kernel_output = 0.0;
        
        for (int j = 0; j < weights.size(); ++j) {
            kernel_output += rbf_kernel(sample, training_data.row(j), gamma) * weights(j);
        }
        
        return (kernel_output > 0) ? 1 : -1;
    }

    VectorXi predict(const MatrixXd& test_data, const MatrixXd& training_data) {
        VectorXi predictions(test_data.rows());
        
        for (int i = 0; i < test_data.rows(); ++i) {
            predictions(i) = predict_single(test_data.row(i), training_data);
        }
        
        return predictions;
    }

    double calculate_accuracy(const MatrixXd& test_data, const VectorXd& true_labels, const MatrixXd& training_data) {
        VectorXi predictions = predict(test_data, training_data);
        int correct = 0;
        
        for (int i = 0; i < test_data.rows(); ++i) {
            if (predictions(i) == true_labels(i)) correct++;
        }
        
        return (double)correct / test_data.rows();
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
    
    // Create and train Kernel SVM
    KernelSVM svm(0.01, 0.1, 50, 30, 1e-5);
    svm.fit(train_data, train_labels);
    
    // Test accuracy
    double train_accuracy = svm.calculate_accuracy(train_data, train_labels, train_data);
    double test_accuracy = svm.calculate_accuracy(test_data, test_labels, train_data);
    
    std::cout << "Training accuracy: " << train_accuracy * 100 << "%" << std::endl;
    std::cout << "Test accuracy: " << test_accuracy * 100 << "%" << std::endl;
    
    // Test single prediction
    VectorXd sample = test_data.row(0);
    int prediction = svm.predict_single(sample, train_data);
    std::cout << "Single prediction for first test sample: " << prediction 
              << " (actual: " << test_labels(0) << ")" << std::endl;
    
    return 0;
}