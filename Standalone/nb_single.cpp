// Pure Naive Bayes Implementation - No networking, no federated logic
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;

struct NaiveBayesStats {
    std::vector<VectorXd> means;
    std::vector<VectorXd> variances;
    std::vector<double> priors;
    std::vector<int> sample_counts;
};

class NaiveBayes {
private:
    int num_classes;
    int num_features;
    int batch_size;
    int num_epochs;
    NaiveBayesStats stats;
    
public:
    NaiveBayes(int epochs = 50, int batch = 512)
        : num_epochs(epochs), batch_size(batch) {}

    // Compute class-wise statistics for a given batch
    NaiveBayesStats compute_class_statistics(const MatrixXd& batch_data, const VectorXd& batch_labels, int total_samples, int num_classes) {
        NaiveBayesStats batch_stats;
        batch_stats.means.resize(num_classes, VectorXd::Zero(batch_data.cols()));
        batch_stats.variances.resize(num_classes, VectorXd::Zero(batch_data.cols()));
        batch_stats.priors.resize(num_classes, 0.0);

        for (int c = 0; c < num_classes; ++c) {
            std::vector<int> class_indices;
            for (int i = 0; i < batch_labels.size(); ++i) {
                if (batch_labels[i] == c) {
                    class_indices.push_back(i);
                }
            }

            if (!class_indices.empty()) {
                MatrixXd class_data(class_indices.size(), batch_data.cols());
                for (size_t i = 0; i < class_indices.size(); ++i) {
                    class_data.row(i) = batch_data.row(class_indices[i]);
                }

                batch_stats.means[c] = class_data.colwise().mean();
                batch_stats.variances[c] = (class_data.rowwise() - batch_stats.means[c].transpose()).array().square().colwise().mean();
                batch_stats.priors[c] = static_cast<double>(class_data.rows()) / total_samples;
            } else {
                batch_stats.means[c].setZero();
                batch_stats.variances[c].setOnes();
                batch_stats.priors[c] = 0.0;
            }
        }

        return batch_stats;
    }

    void aggregate_statistics(int class_index, const VectorXd& batch_means, const VectorXd& batch_variances, int batch_size) {
        if (stats.sample_counts[class_index] == 0) {
            stats.means[class_index] = batch_means;
            stats.variances[class_index] = batch_variances;
            stats.sample_counts[class_index] = batch_size;
            stats.priors[class_index] = static_cast<double>(batch_size);
        } else {
            int total_samples = stats.sample_counts[class_index] + batch_size;

            // Update means
            VectorXd delta_means = batch_means - stats.means[class_index];
            stats.means[class_index] += (batch_size * delta_means.array()).matrix() / total_samples;

            // Update variances
            VectorXd combined_variances = (((stats.sample_counts[class_index] - 1) * stats.variances[class_index].array() +
                                            (batch_size - 1) * batch_variances.array() +
                                            (stats.sample_counts[class_index] * batch_size * delta_means.array().square()) /
                                                total_samples) /
                                           (total_samples - 1))
                                              .matrix();
            stats.variances[class_index] = combined_variances;
            stats.sample_counts[class_index] = total_samples;
            stats.priors[class_index] = static_cast<double>(total_samples);
        }
    }

    void fit(const MatrixXd& data, const VectorXd& labels) {
        std::cout << "Training Naive Bayes..." << std::endl;
        
        // Determine number of classes and features
        num_classes = static_cast<int>(*std::max_element(labels.data(), labels.data() + labels.size())) + 1;
        num_features = data.cols();
        
        // Initialize statistics
        stats.means.resize(num_classes, VectorXd::Zero(num_features));
        stats.variances.resize(num_classes, VectorXd::Ones(num_features));
        stats.priors.resize(num_classes, 0.0);
        stats.sample_counts.resize(num_classes, 0);

        int num_batches = data.rows() / batch_size + (data.rows() % batch_size != 0);
        int total_samples = data.rows();

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::cout << "Epoch " << epoch + 1 << std::endl;
            
            for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
                int start_row = batch_idx * batch_size;
                int end_row = std::min(static_cast<int>(data.rows()), start_row + batch_size);

                MatrixXd batch_data = data.block(start_row, 0, end_row - start_row, data.cols());
                VectorXd batch_labels = labels.segment(start_row, end_row - start_row);

                NaiveBayesStats batch_stats = compute_class_statistics(batch_data, batch_labels, total_samples, num_classes);

                // Aggregate statistics for each class
                for (int c = 0; c < num_classes; ++c) {
                    int class_batch_size = static_cast<int>(batch_stats.priors[c] * total_samples);
                    if (class_batch_size > 0) {
                        aggregate_statistics(c, batch_stats.means[c], batch_stats.variances[c], class_batch_size);
                    }
                }
            }
        }
        
        // Normalize priors
        double total_prior = 0.0;
        for (int c = 0; c < num_classes; ++c) {
            total_prior += stats.priors[c];
        }
        for (int c = 0; c < num_classes; ++c) {
            stats.priors[c] /= total_prior;
        }
        
        std::cout << "Training completed" << std::endl;
    }

    int predict_single(const VectorXd& sample) {
        std::vector<double> log_probs(num_classes, 0.0);

        for (int c = 0; c < num_classes; ++c) {
            log_probs[c] = std::log(stats.priors[c]);

            for (int i = 0; i < sample.size(); ++i) {
                double x = sample[i];
                if (stats.variances[c][i] > 0) {
                    double log_likelihood = -0.5 * std::log(2 * M_PI * stats.variances[c][i]) -
                                            (std::pow(x - stats.means[c][i], 2) / (2 * stats.variances[c][i]));
                    log_probs[c] += log_likelihood;
                }
            }
        }

        return std::distance(log_probs.begin(), std::max_element(log_probs.begin(), log_probs.end()));
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

    void predict_samples(const MatrixXd& test_data) {
        std::cout << "Starting predictions for test data..." << std::endl;

        for (int i = 0; i < test_data.rows(); ++i) {
            VectorXd sample = test_data.row(i);
            int predicted_class = predict_single(sample);
            std::cout << "Sample " << i + 1 << " predicted class: " << predicted_class << std::endl;
        }

        std::cout << "Predictions completed." << std::endl;
    }

    NaiveBayesStats get_stats() const { return stats; }
};

// Simple data generator for testing
std::pair<MatrixXd, VectorXd> generate_test_data(int n_samples = 1000, int n_features = 10, int n_classes = 3) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> feature_dist(0.0, 1.0);
    std::uniform_int_distribution<> class_dist(0, n_classes - 1);
    
    MatrixXd data(n_samples, n_features);
    VectorXd labels(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            data(i, j) = feature_dist(gen);
        }
        // Simple rule based on first few features
        if (data(i, 0) + data(i, 1) > 1.0) {
            labels(i) = 0;
        } else if (data(i, 0) + data(i, 1) < -1.0) {
            labels(i) = 1;
        } else {
            labels(i) = 2;
        }
    }
    
    return {data, labels};
}

int main() {
    // Generate test data
    auto [data, labels] = generate_test_data(1000, 10, 3);
    
    // Split into train and test
    int train_size = 800;
    MatrixXd train_data = data.topRows(train_size);
    VectorXd train_labels = labels.head(train_size);
    MatrixXd test_data = data.bottomRows(data.rows() - train_size);
    VectorXd test_labels = labels.tail(labels.size() - train_size);
    
    // Create and train Naive Bayes
    NaiveBayes nb(20, 512);  // 20 epochs, batch size 512
    nb.fit(train_data, train_labels);
    
    // Test accuracy
    double train_accuracy = nb.calculate_accuracy(train_data, train_labels);
    double test_accuracy = nb.calculate_accuracy(test_data, test_labels);
    
    std::cout << "Training accuracy: " << train_accuracy * 100 << "%" << std::endl;
    std::cout << "Test accuracy: " << test_accuracy * 100 << "%" << std::endl;
    
    // Test single prediction
    VectorXd sample = test_data.row(0);
    int prediction = nb.predict_single(sample);
    std::cout << "Single prediction for first test sample: " << prediction 
              << " (actual: " << test_labels(0) << ")" << std::endl;
    
    // Show class statistics
    auto stats = nb.get_stats();
    std::cout << "\nClass Statistics:" << std::endl;
    for (int c = 0; c < 3; ++c) {
        std::cout << "Class " << c << " - Prior: " << stats.priors[c] 
                  << ", Sample count: " << stats.sample_counts[c] << std::endl;
    }
    
    return 0;
}