#include <iostream>
#include <vector>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include <mutex>

using namespace Eigen;
using boost::asio::ip::tcp;

// Global variables for aggregated statistics
std::vector<VectorXd> global_means;
std::vector<VectorXd> global_variances;
std::vector<double> global_priors;
std::vector<int> global_sample_counts;
std::mutex model_mutex;

void aggregate_statistics(int class_index, const VectorXd& batch_means, const VectorXd& batch_variances, int batch_size) {
    std::lock_guard<std::mutex> lock(model_mutex);

    if (global_sample_counts[class_index] == 0) {
        global_means[class_index] = batch_means;
        global_variances[class_index] = batch_variances;
        global_sample_counts[class_index] = batch_size;
        global_priors[class_index] = static_cast<double>(batch_size);
    } else {
        int total_samples = global_sample_counts[class_index] + batch_size;

        // Update means
        VectorXd delta_means = batch_means - global_means[class_index];
        global_means[class_index] += (batch_size * delta_means.array()).matrix() / total_samples;

        // Update variances
        VectorXd combined_variances = (((global_sample_counts[class_index] - 1) * global_variances[class_index].array() +
                                        (batch_size - 1) * batch_variances.array() +
                                        (global_sample_counts[class_index] * batch_size * delta_means.array().square()) /
                                            total_samples) /
                                       (total_samples - 1))
                                          .matrix();
        global_variances[class_index] = combined_variances;
        global_sample_counts[class_index] = total_samples;
        global_priors[class_index] = static_cast<double>(total_samples);
    }
}

void process_batches(tcp::socket& socket) {
    try {
        while (socket.is_open()) {
            int total_samples;
            int num_classes;
            int num_features;

            // Read batch-specific metadata for each batch
            boost::asio::read(socket, boost::asio::buffer(&total_samples, sizeof(int)));
            boost::asio::read(socket, boost::asio::buffer(&num_classes, sizeof(int)));
            boost::asio::read(socket, boost::asio::buffer(&num_features, sizeof(int)));
            std::cout << "[DEBUG] Total samples in dataset: " << total_samples << std::endl;
            std::cout<<"Number of features "<<num_features<<std::endl;

            if (global_means.empty()) {
                // Initialize global structures
                global_means.resize(num_classes, VectorXd::Zero(num_features));
                global_variances.resize(num_classes, VectorXd::Ones(num_features));
                global_priors.resize(num_classes, 0.0);
                global_sample_counts.resize(num_classes, 0);
            }

            for (int c = 0; c < num_classes; ++c) {
                double prior;
                VectorXd mean(num_features);
                VectorXd variance(num_features);

                // Read class-specific statistics
                boost::asio::read(socket, boost::asio::buffer(&prior, sizeof(double)));
                boost::asio::read(socket, boost::asio::buffer(mean.data(), num_features * sizeof(double)));
                boost::asio::read(socket, boost::asio::buffer(variance.data(), num_features * sizeof(double)));

                // Update global statistics
                aggregate_statistics(c, mean, variance, static_cast<int>(prior * total_samples));
                
            }

            for (int c = 0; c < num_classes; ++c) {
                // Send updated global statistics back to the client
                boost::asio::write(socket, boost::asio::buffer(global_means[c].data(), num_features * sizeof(double)));
                boost::asio::write(socket, boost::asio::buffer(global_variances[c].data(), num_features * sizeof(double)));
                boost::asio::write(socket, boost::asio::buffer(&global_priors[c], sizeof(double)));
            }

            std::cout << "[DEBUG] Processed batch for Class 0 and Class 1." << std::endl;

            
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception during batch processing: " << e.what() << std::endl;
    }
}


void handle_client(tcp::socket socket) {
    process_batches(socket);
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 8080));

        std::cout << "[DEBUG] Server started. Waiting for clients..." << std::endl;

        while (true) {
            tcp::socket socket(io_context);
            acceptor.accept(socket);

            std::cout << "[DEBUG] Client connected." << std::endl;

            // Use std::move to transfer ownership of the socket
            std::thread(handle_client, std::move(socket)).detach();
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in server: " << e.what() << std::endl;
    }

    return 0;
}

