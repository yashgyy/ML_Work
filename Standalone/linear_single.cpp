// Pure Linear Regression Implementation - No networking, no federated logic
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;

class LinearRegression {
private:
    VectorXd weights;
    double learning_rate;
    int max_epochs;
    int batch_size;
    double regularization_strength;
    
public:
    LinearRegression(double lr = 0.005, int epochs = 50, int batch = 512, double reg = 0.001)
        : learning_rate(lr), max_epochs(epochs), batch_size(batch), regularization_strength(reg) {}

    // Mean Squared Error (MSE) Loss Function
    double compute_mse(const MatrixXd& data, const VectorXd& labels, const VectorXd& weights) {
        VectorXd predictions = data * weights;
        VectorXd errors = labels - predictions;
        return (errors.squaredNorm() / labels.size());
    }

    // Standardize data (zero mean, unit variance)
    void standardize_data(MatrixXd& data, VectorXd& means, VectorXd& stddevs) {
        means = data.colwise().mean();
        stddevs = ((data.rowwise() - means.transpose()).array().square().colwise().mean()).sqrt();
        
        for (int i = 0; i < data.cols(); ++i) {
            data.col(i) = (data.col(i).array() - means(i)) / (stddevs(i) + 1e-8);  // Avoid divide-by-zero
        }
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

                // Compute the gradient using MSE loss
                for (int j = 0; j < current_batch_size; ++j) {
                    VectorXd xi = data.row(indices[i + j]);
                    double yi = labels(indices[i + j]);
                    double prediction = xi.dot(weights);
                    gradient += -2 * xi * (yi - prediction) + regularization_strength * weights;
                }

                gradient /= current_batch_size;  // Normalize gradient

                // Apply gradient update
                weights -= learning_rate * gradient;

                // Compute and display MSE loss after update
                if (i % (batch_size * 10) == 0) {  // Display every 10 batches
                    double mse_loss = compute_mse(data, labels, weights);
                    std::cout << "Epoch " << epoch + 1 << ", Batch " << (i / batch_size) + 1
                              << " - MSE Loss: " << mse_loss << std::endl;
                }
            }
        }
    }

    void fit(const MatrixXd& data, const VectorXd& labels) {
        std::cout << "Training Linear Regression..." << std::endl;
        
        // Make a copy of data for standardization
        MatrixXd train_data = data;
        VectorXd means, stddevs;
        
        // Standardize the training data
        standardize_data(train_data, means, stddevs);
        
        // Initialize weights randomly
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 0.01);
        weights = VectorXd::Zero(train_data.cols()).unaryExpr([&](double) { return d(gen); });
        
        // Train using batch processing
        train_batch(train_data, labels);
        
        std::cout << "Training completed" << std::endl;
    }

    double predict_single(const VectorXd& sample) {
        return sample.dot(weights);
    }

    VectorXd predict(const MatrixXd& data) {
        return data * weights;
    }

    double calculate_mse(const MatrixXd& data, const VectorXd& true_labels) {
        return compute_mse(data, true_labels, weights);
    }

    // Calculate R-squared score
    double calculate_r2(const MatrixXd& data, const VectorXd& true_labels) {
        VectorXd predictions = predict(data);
        double ss_res = (true_labels - predictions).squaredNorm();
        double ss_tot = (true_labels.array() - true_labels.mean()).square().sum();
        return 1.0 - (ss_res / ss_tot);
    }

    // Calculate Mean Absolute Error (MAE)
    double calculate_mae(const MatrixXd& data, const VectorXd& true_labels) {
        VectorXd predictions = predict(data);
        return (true_labels - predictions).cwiseAbs().mean();
    }

    // Calculate Root Mean Squared Error (RMSE)
    double calculate_rmse(const MatrixXd& data, const VectorXd& true_labels) {
        return std::sqrt(calculate_mse(data, true_labels));
    }

    VectorXd get_weights() const { return weights; }
    void set_weights(const VectorXd& new_weights) { weights = new_weights; }
};

// Simple data generator for testing
std::pair<MatrixXd, VectorXd> generate_test_data(int n_samples = 1000, int n_features = 10) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> feature_dist(0.0, 1.0);
    std::normal_distribution<> noise_dist(0.0, 0.1);
    
    MatrixXd data(n_samples, n_features);
    VectorXd labels(n_samples);
    
    // Generate true weights for synthetic data
    VectorXd true_weights = VectorXd::Random(n_features);
    
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            data(i, j) = feature_dist(gen);
        }
        // Generate labels using true weights plus noise
        labels(i) = data.row(i).dot(true_weights) + noise_dist(gen);
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
    
    // Create and train Linear Regression
    LinearRegression lr(0.005, 50, 512, 0.001);
    lr.fit(train_data, train_labels);
    
    // Test performance
    double train_mse = lr.calculate_mse(train_data, train_labels);
    double test_mse = lr.calculate_mse(test_data, test_labels);
    double train_r2 = lr.calculate_r2(train_data, train_labels);
    double test_r2 = lr.calculate_r2(test_data, test_labels);
    double train_mae = lr.calculate_mae(train_data, train_labels);
    double test_mae = lr.calculate_mae(test_data, test_labels);
    double train_rmse = lr.calculate_rmse(train_data, train_labels);
    double test_rmse = lr.calculate_rmse(test_data, test_labels);
    
    std::cout << "\n=== Model Performance ===" << std::endl;
    std::cout << "Training MSE: " << train_mse << std::endl;
    std::cout << "Test MSE: " << test_mse << std::endl;
    std::cout << "Training R²: " << train_r2 << std::endl;
    std::cout << "Test R²: " << test_r2 << std::endl;
    std::cout << "Training MAE: " << train_mae << std::endl;
    std::cout << "Test MAE: " << test_mae << std::endl;
    std::cout << "Training RMSE: " << train_rmse << std::endl;
    std::cout << "Test RMSE: " << test_rmse << std::endl;
    
    // Test single prediction
    VectorXd sample = test_data.row(0);
    double prediction = lr.predict_single(sample);
    std::cout << "\nSingle prediction for first test sample: " << prediction 
              << " (actual: " << test_labels(0) << ")" << std::endl;
    
    // Show some predictions vs actual
    VectorXd predictions = lr.predict(test_data.topRows(5));
    std::cout << "\nFirst 5 predictions vs actual:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "Sample " << i + 1 << " - Predicted: " << predictions(i) 
                  << ", Actual: " << test_labels(i) 
                  << ", Error: " << std::abs(predictions(i) - test_labels(i)) << std::endl;
    }
    
    return 0;
}