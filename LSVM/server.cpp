// Refactored Federated Linear SVM Server (based on K_Server structure)
#include <iostream>
#include <vector>
#include <thread>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include <mutex>
#include <numeric>
#include "data_loader.cpp"

using namespace Eigen;
using boost::asio::ip::tcp;

std::mutex model_mutex;
VectorXd global_weights;
VectorXd total_gradients;
std::vector<int> client_data_sizes;
int client_count = 0;

const int BATCH_SIZE = 100;

void apply_gradient_update(const VectorXd& batch_gradient, int batch_size) {
    std::lock_guard<std::mutex> lock(model_mutex);

    if (global_weights.size() == 0) {
        global_weights = VectorXd::Zero(batch_gradient.size());
        total_gradients = VectorXd::Zero(batch_gradient.size());
    }

    total_gradients += batch_gradient * batch_size;
    client_data_sizes.push_back(batch_size);
    client_count++;

    int total_points = std::accumulate(client_data_sizes.begin(), client_data_sizes.end(), 0);
    global_weights -= (total_gradients / total_points);

    std::cout << "[DEBUG] Updated global weights: " << global_weights.transpose() << std::endl;
    total_gradients.setZero();
}

void handle_client(tcp::socket socket) {
    try {
        std::cout << "[DEBUG] Client connected." << std::endl;

        while (true) {
            int batch_size = 0, vector_size = 0;
            boost::asio::read(socket, boost::asio::buffer(&batch_size, sizeof(int)));
            boost::asio::read(socket, boost::asio::buffer(&vector_size, sizeof(int)));

            if (vector_size <= 0 || vector_size > 1e7) {
                std::cerr << "[ERROR] Invalid vector size." << std::endl;
                return;
            }

            VectorXd batch_gradient = VectorXd::Zero(vector_size);
            int received = 0;
            while (received < vector_size) {
                int chunk_size = std::min(BATCH_SIZE, vector_size - received);
                boost::asio::read(socket, boost::asio::buffer(batch_gradient.data() + received, chunk_size * sizeof(double)));
                received += chunk_size;
            }

            apply_gradient_update(batch_gradient, batch_size);

            boost::asio::write(socket, boost::asio::buffer(global_weights.data(), global_weights.size() * sizeof(double)));
            std::cout << "[DEBUG] Sent global model to client." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in handle_client: " << e.what() << std::endl;
    }
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 8080));

        std::cout << "[INFO] Server started. Waiting for clients...\n";

        while (true) {
            tcp::socket socket(io_context);
            acceptor.accept(socket);
            std::thread(handle_client, std::move(socket)).detach();
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in server: " << e.what() << std::endl;
    }
    return 0;
}
