#include <iostream>
#include <vector>
#include <thread>
#include <fstream>
#include <boost/asio.hpp>
#include <mutex>
#include <Eigen/Dense>

using namespace Eigen;
using boost::asio::ip::tcp;

std::mutex model_mutex;  // Mutex for thread-safe operations
VectorXd global_weights;  // Store the global model
VectorXd total_weights;   // Accumulate weights for averaging
int client_count = 0;     // Track the number of connected clients

const int BATCH_SIZE = 100;  // Batch size for receiving updates

// Federated Aggregation of Local Models
void aggregate_model(const VectorXd& local_update) {
    std::lock_guard<std::mutex> lock(model_mutex);  // Ensure thread safety

    if (total_weights.size() == 0) {
        total_weights = VectorXd::Zero(local_update.size());
    }

    total_weights += local_update;
    client_count++;

    // Update the global weights by averaging
    global_weights = total_weights / client_count;

    std::cout << "[DEBUG] Aggregated global weights (first 10): "
              << global_weights.head(10).transpose() << std::endl;
}

void handle_client(tcp::socket socket) {
    try {
        std::cout << "[DEBUG] Handling new client connection." << std::endl;

        int vector_size = 0;
while (true) {
    int vector_size = 0;
    boost::system::error_code error;

    // Read vector size before processing each epoch/batch
    boost::asio::read(socket, boost::asio::buffer(&vector_size, sizeof(int)), error);

    if (error == boost::asio::error::eof) {
        std::cout << "[INFO] Client disconnected gracefully. Ending session." << std::endl;
        break;
    } else if (error) {
        std::cerr << "[ERROR] Failed to receive vector size: " << error.message() << std::endl;
        return;
    }

    std::cout << "[DEBUG] Received vector size: " << vector_size << std::endl;

    // Validate the received vector size
    if (vector_size <= 0 || vector_size > 1e7) {
        std::cerr << "[ERROR] Invalid vector size received: " << vector_size << std::endl;
        break;
    }

    VectorXd local_update = VectorXd::Zero(vector_size);
    int received = 0;

    // Receive the complete vector data in batches
    while (received < vector_size) {
        int batch_size = std::min(BATCH_SIZE, vector_size - received);
        boost::asio::read(socket, boost::asio::buffer(local_update.data() + received, batch_size * sizeof(double)), error);

        if (error == boost::asio::error::eof) {
            std::cerr << "[ERROR] Client disconnected during data batch." << std::endl;
            return;
        } else if (error) {
            std::cerr << "[ERROR] Error while reading data batch: " << error.message() << std::endl;
            return;
        }

        received += batch_size;
        std::cout << "[DEBUG] Received batch of size: " << batch_size
                  << " | Total received: " << received << "/" << vector_size << std::endl;
    }

    std::cout << "[DEBUG] Received complete vector. Total size: " << received << std::endl;

    // Aggregate the local update into the global model
    aggregate_model(local_update);

    // Send updated global weights back to the client
    boost::asio::write(socket, boost::asio::buffer(global_weights.data(), global_weights.size() * sizeof(double)));
    std::cout << "[DEBUG] Sent updated global model to client." << std::endl;
}

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in handle_client: " << e.what() << std::endl;
    }
}


// Main function to run the server
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
