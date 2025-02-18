#include <iostream>
#include <vector>
#include <thread>
#include <boost/asio.hpp>
#include <mutex>
#include <Eigen/Dense>
//g++ server_continue.cpp -o server  -I /usr/include/eigen3

using namespace Eigen;
using boost::asio::ip::tcp;

std::mutex model_mutex;
VectorXd global_weights;

VectorXd total_weights = VectorXd::Zero(global_weights.size());  // Initialize to zero
int client_count = 0;

void aggregate_model(const VectorXd& local_update) {
    std::lock_guard<std::mutex> lock(model_mutex);
    total_weights += local_update;
    client_count++;
    global_weights = total_weights / client_count;
    std::cout << "[DEBUG] Aggregated global model weights: " << global_weights.transpose() << std::endl;
}

void handle_client(tcp::socket socket) {
    try {
        std::cout << "[DEBUG] Handling new client connection." << std::endl;

        // Read the size of the incoming vector first
        int vector_size;
        boost::asio::read(socket, boost::asio::buffer(&vector_size, sizeof(int)));

        // Resize the local update vector based on the size received
        VectorXd local_update(vector_size);

        std::cout << "[DEBUG] Waiting to receive data from client..." << std::endl;

        // Read local update from client
        boost::asio::read(socket, boost::asio::buffer(local_update.data(), local_update.size() * sizeof(double)));
        std::cout << "[DEBUG] Received local update of size: " << local_update.size() << std::endl;

        // Initialize global_weights and total_weights if not already initialized
        if (global_weights.size() == 0) {
            global_weights = VectorXd::Zero(local_update.size());  // Initialize to correct size
            total_weights = VectorXd::Zero(local_update.size());   // Initialize to the same size
            std::cout << "[DEBUG] Initialized global_weights and total_weights with size: " << global_weights.size() << std::endl;
        }

        // Ensure sizes match before aggregating
        if (global_weights.size() == local_update.size()) {
            aggregate_model(local_update);
        } else {
            std::cerr << "[ERROR] Size mismatch between global model and local update!" << std::endl;
        }

        // Send updated global model back to client
        boost::asio::write(socket, boost::asio::buffer(global_weights.data(), global_weights.size() * sizeof(double)));
        std::cout << "[DEBUG] Sent updated global model to client." << std::endl;

    } catch (std::exception& e) {
        std::cerr << "[ERROR] Exception in handle_client: " << e.what() << std::endl;
    }
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 8080));

        std::cout << "[DEBUG] Server started. Waiting for clients on port 8080..." << std::endl;

        while (true) {
            tcp::socket socket(io_context);
            std::cout << "[DEBUG] Waiting for new client connection..." << std::endl;
            acceptor.accept(socket);
            std::cout << "[DEBUG] Client connected." << std::endl;

            // Start a thread to handle each client
            std::thread(handle_client, std::move(socket)).detach();
        }
    } catch (std::exception& e) {
        std::cerr << "[ERROR] Exception in main: " << e.what() << std::endl;
    }

    return 0;
}
