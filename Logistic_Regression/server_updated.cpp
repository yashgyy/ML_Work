// Updated server.cpp
#include <iostream>
#include <vector>
#include <thread>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include <mutex>
#include "data_loader.cpp" // Include the data loader

using namespace Eigen;
using boost::asio::ip::tcp;

std::mutex model_mutex;  // Mutex for thread-safe operations
VectorXd global_weights;  // Global model weights
VectorXd total_weights;   // Aggregated weights for averaging
int client_count = 0;     // Number of connected clients

const int BATCH_SIZE = 100;  // Batch size for receiving updates

// Function to aggregate local updates into the global model
void aggregate_model(const VectorXd& local_update) {
    std::lock_guard<std::mutex> lock(model_mutex);

    if (client_count == 0) {
        global_weights = VectorXd::Zero(local_update.size());
        total_weights = VectorXd::Zero(local_update.size());
    }

    total_weights += local_update;
    client_count++;

    global_weights = total_weights / client_count;

    if (!global_weights.allFinite()) {
        std::cerr << "[ERROR] Global weights contain NaN or inf values. Resetting to zero." << std::endl;
        global_weights.setZero();
    }

    std::cout << "[DEBUG] Aggregated global weights (first 10): "
              << global_weights.head(10).transpose() << std::endl;
}

// Function to handle communication with a client
void handle_client(tcp::socket socket) {
    try {
        std::cout << "[DEBUG] Handling new client connection." << std::endl;

        // Read the size of the incoming weight vector
        int vector_size = 0;
        boost::asio::read(socket, boost::asio::buffer(&vector_size, sizeof(vector_size)));
        std::cout << "[DEBUG] Received vector size: " << vector_size << std::endl;

        // Validate vector size
        if (vector_size <= 0 || vector_size > 1e7) {
            std::cerr << "[ERROR] Invalid vector size received: " << vector_size << std::endl;
            return;
        }

        while (true) {
            VectorXd local_update = VectorXd::Zero(vector_size);
            int received = 0;

            std::cout << "[DEBUG] Receiving data in batches..." << std::endl;

            while (received < vector_size) {
                int batch_size = std::min(BATCH_SIZE, vector_size - received);
                boost::asio::read(socket, boost::asio::buffer(
                    local_update.data() + received, batch_size * sizeof(double)));

                received += batch_size;
            }

            std::cout << "[DEBUG] First 10 weights received: " 
                      << local_update.head(10).transpose() << std::endl;

            aggregate_model(local_update);

            boost::asio::write(socket, boost::asio::buffer(
                global_weights.data(), global_weights.size() * sizeof(double)));

            std::cout << "[DEBUG] Sent updated global model to client." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in handle_client: " << e.what() << std::endl;
    }
}

// Main function to start the server
int main() {
    try {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 8080));

        std::cout << "[DEBUG] Server started. Waiting for clients on port 8080..." << std::endl;

        while (true) {
            tcp::socket socket(io_context);
            acceptor.accept(socket);
            std::cout << "[DEBUG] Client connected." << std::endl;

            std::thread(handle_client, std::move(socket)).detach();
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in main: " << e.what() << std::endl;
    }

    return 0;
}
