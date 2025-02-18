#include <iostream>
#include <vector>
#include <thread>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include <mutex>
#include <numeric>

using namespace Eigen;
using boost::asio::ip::tcp;

std::mutex model_mutex;  // Mutex for thread-safe operations
VectorXd global_weights;  // Store the global model
VectorXd total_gradients; // Accumulate gradients for averaging
int client_count = 0;     // Track the number of connected clients
std::vector<int> client_data_sizes; // Store data sizes per client

const int BATCH_SIZE = 100;  // Batch size for receiving updates

// **Apply Gradient Updates to Global Model**
void apply_gradient_update(const VectorXd& batch_gradient, int batch_size) {
    std::lock_guard<std::mutex> lock(model_mutex);  // Ensure thread safety

    if (client_count == 0) {
        total_gradients = VectorXd::Zero(batch_gradient.size());
        global_weights = VectorXd::Zero(batch_gradient.size()); // Initialize weights
    }

    total_gradients += batch_gradient * batch_size;  // Accumulate batch-wise gradients
    client_data_sizes.push_back(batch_size);
    client_count++;

    // Compute weighted average update
    int total_data_points = std::accumulate(client_data_sizes.begin(), client_data_sizes.end(), 0);
    global_weights -= (total_gradients / total_data_points); // Apply gradient descent step

    std::cout << "[DEBUG] Updated global weights (first 10 values): "
              << global_weights.head(10).transpose() << std::endl;

    // **Reset total_gradients after applying update**
    total_gradients.setZero();
}

// **Handles communication with a client**
void handle_client(tcp::socket socket) {
    try {
        std::cout << "[DEBUG] New client connected." << std::endl;

        while (true) {
            int batch_size = 0, vector_size = 0;

            // **Read batch size and vector size from client**
            boost::asio::read(socket, boost::asio::buffer(&batch_size, sizeof(int)));
            boost::asio::read(socket, boost::asio::buffer(&vector_size, sizeof(int)));

            if (vector_size <= 0 || vector_size > 1e7) {
                std::cerr << "[ERROR] Invalid vector size received: " << vector_size << std::endl;
                return;
            }

            std::cout << "[DEBUG] Receiving batch of size: " << batch_size << ", Vector size: " << vector_size << std::endl;

            VectorXd batch_gradient = VectorXd::Zero(vector_size);
            int received = 0;

            while (received < vector_size) {
                int chunk_size = std::min(BATCH_SIZE, vector_size - received);
                boost::asio::read(socket, boost::asio::buffer(batch_gradient.data() + received, chunk_size * sizeof(double)));
                received += chunk_size;
            }

            // **Apply the received gradient update**
            apply_gradient_update(batch_gradient, batch_size);

            // **Send updated global model to client after every batch**
            boost::asio::write(socket, boost::asio::buffer(global_weights.data(), global_weights.size() * sizeof(double)));

            std::cout << "[DEBUG] Sent updated global model to client after batch." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in handle_client: " << e.what() << std::endl;
    }
}

// **Main function to run the server**
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
        std::cerr << "[ERROR] Exception in server: " << e.what() << std::endl;
    }

    return 0;
}
