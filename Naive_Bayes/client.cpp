#include <iostream>
#include <vector>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include "data_loader.cpp"
//#include "data_loader_susy.cpp"
//#include "data_loader_higgs.cpp"

using namespace Eigen;
using boost::asio::ip::tcp;

const int COMPUTE_BATCH_SIZE = 100;

struct NaiveBayesBatchStats {
    std::vector<VectorXd> means;
    std::vector<VectorXd> variances;
    std::vector<double> priors;
};

// Compute class-wise statistics for a given batch
NaiveBayesBatchStats compute_class_statistics(const MatrixXd& batch_data, const VectorXd& batch_labels, int total_samples, int num_classes) {
    NaiveBayesBatchStats stats;
    stats.means.resize(num_classes, VectorXd::Zero(batch_data.cols()));
    stats.variances.resize(num_classes, VectorXd::Zero(batch_data.cols()));
    stats.priors.resize(num_classes, 0.0);

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

            stats.means[c] = class_data.colwise().mean();
            stats.variances[c] = (class_data.rowwise() - stats.means[c].transpose()).array().square().colwise().mean();
            stats.priors[c] = static_cast<double>(class_data.rows()) / total_samples;
        } else {
            stats.means[c].setZero();
            stats.variances[c].setOnes();
            stats.priors[c] = 0.0;
        }
    }

    return stats;
}

int predict(const VectorXd& sample, const std::vector<VectorXd>& means, const std::vector<VectorXd>& variances, const std::vector<double>& priors, int num_classes) {
    std::vector<double> log_probs(num_classes, 0.0);

    for (int c = 0; c < num_classes; ++c) {
        log_probs[c] = std::log(priors[c]);

        for (int i = 0; i < sample.size(); ++i) {
            double x = sample[i];
            if (variances[c][i] > 0) {
                double log_likelihood = -0.5 * std::log(2 * M_PI * variances[c][i]) -
                                        (std::pow(x - means[c][i], 2) / (2 * variances[c][i]));
                log_probs[c] += log_likelihood;
            }
        }
    }

    return std::distance(log_probs.begin(), std::max_element(log_probs.begin(), log_probs.end()));
}

void predict_samples(const MatrixXd& test_data, const std::vector<VectorXd>& means, const std::vector<VectorXd>& variances, const std::vector<double>& priors, int num_classes) {
    std::cout << "[DEBUG] Starting predictions for test data..." << std::endl;

    for (int i = 0; i < test_data.rows(); ++i) {
        VectorXd sample = test_data.row(i);
        int predicted_class = predict(sample, means, variances, priors, num_classes);

        std::cout << "Sample " << i + 1 << " predicted class: " << predicted_class << std::endl;
    }

    std::cout << "[DEBUG] Predictions completed." << std::endl;
}

void send_batches_and_receive_updates(tcp::socket& socket, const MatrixXd& local_data, const VectorXd& local_labels, int num_epochs, int num_classes) {
    //int num_classes = *std::max_element(local_labels.data(), local_labels.data() + local_labels.size()) + 1;
    int num_batches = local_data.rows() / COMPUTE_BATCH_SIZE + (local_data.rows() % COMPUTE_BATCH_SIZE != 0);
    std::vector<VectorXd> updated_means(num_classes);
    std::vector<VectorXd> updated_variances(num_classes);
    std::vector<double> updated_priors(num_classes);

    try {
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
                int start_row = batch_idx * COMPUTE_BATCH_SIZE;
                int end_row = std::min(static_cast<int>(local_data.rows()), start_row + COMPUTE_BATCH_SIZE);

                MatrixXd batch_data = local_data.block(start_row, 0, end_row - start_row, local_data.cols());
                VectorXd batch_labels = local_labels.segment(start_row, end_row - start_row);

                NaiveBayesBatchStats batch_stats = compute_class_statistics(batch_data, batch_labels, local_data.rows(), num_classes);
                int num_features = batch_data.cols();
                int total_samples = local_data.rows();

                boost::asio::write(socket, boost::asio::buffer(&total_samples, sizeof(int)));
                boost::asio::write(socket, boost::asio::buffer(&num_classes, sizeof(int)));
                boost::asio::write(socket, boost::asio::buffer(&num_features, sizeof(int)));

                for (int c = 0; c < num_classes; ++c) {
                    boost::asio::write(socket, boost::asio::buffer(&batch_stats.priors[c], sizeof(double)));
                    boost::asio::write(socket, boost::asio::buffer(batch_stats.means[c].data(), num_features * sizeof(double)));
                    boost::asio::write(socket, boost::asio::buffer(batch_stats.variances[c].data(), num_features * sizeof(double)));
                }

                for (int c = 0; c < num_classes; ++c) {
                    updated_means[c].resize(num_features);
                    updated_variances[c].resize(num_features);

                    boost::asio::read(socket, boost::asio::buffer(updated_means[c].data(), num_features * sizeof(double)));
                    boost::asio::read(socket, boost::asio::buffer(updated_variances[c].data(), num_features * sizeof(double)));
                    boost::asio::read(socket, boost::asio::buffer(&updated_priors[c], sizeof(double)));
                }

                std::cout << "[DEBUG] Updated statistics received from server." << std::endl;
                
                // for (int c = 0; c < num_classes; ++c) {
                //     std::cout << "Class " << c << " - Means: " << updated_means[c].transpose()
                //               << ", Variances: " << updated_variances[c].transpose()
                //               << ", Prior: " << updated_priors[c] << std::endl;
                // }
            }
        }

        socket.close();

        std::vector<std::vector<float>> test_features;
        std::vector<int> test_labels;
        load_data("../Datasets/santander-customer-transaction-prediction.csv", test_features, test_labels);
        //load_data("../Datasets/SUSY.csv", test_features, test_labels);
        //load_data("../Datasets/HIGGS.csv", test_features, test_labels);

        MatrixXd test_data(test_features.size(), test_features[0].size());
        for (size_t i = 0; i < test_features.size(); ++i) {
            for (size_t j = 0; j < test_features[i].size(); ++j) {
                test_data(i, j) = test_features[i][j];
            }
        }

        predict_samples(test_data, updated_means, updated_variances, updated_priors, num_classes);

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception during batch sending or receiving: " << e.what() << std::endl;
    }
}

int main() {
    try {
        std::vector<std::vector<float>> features;
        std::vector<int> labels;
        load_data("../Datasets/santander-customer-transaction-prediction.csv", features, labels);
        //load_data("../Datasets/SUSY.csv", features, labels);
        //load_data("../Datasets/HIGGS.csv", features, labels);
        
        MatrixXd local_data(features.size(), features[0].size());
        VectorXd local_labels(labels.size());
        for (size_t i = 0; i < features.size(); ++i) {
            for (size_t j = 0; j < features[i].size(); ++j) {
                local_data(i, j) = features[i][j];
            }
            local_labels(i) = labels[i];
        }

        int num_classes = *std::max_element(labels.begin(), labels.end()) + 1;

        boost::asio::io_context io_context;
        tcp::socket socket(io_context);
        socket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 8080));

        send_batches_and_receive_updates(socket, local_data, local_labels, 1, num_classes);

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in client: " << e.what() << std::endl;
    }

    return 0;
}
