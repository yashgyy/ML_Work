#include <iostream>
#include <vector>
#include <thread>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include <mutex>
#include <numeric>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>


using namespace Eigen;
using boost::asio::ip::tcp;

// g++ server_primary.cpp -o server  -I /usr/include/eigen3

std::mutex model_mutex;  // Mutex for thread-safe operations
VectorXd global_weights;  // Store the global model
VectorXd total_gradients; // Accumulate gradients for averaging
int client_count = 0;     // Track the number of connected clients
std::vector<int> client_data_sizes; // Store data sizes per client
const double LEARNING_RATE = 0.005;  // Learning rate now applied on the server
const int BATCH_SIZE = 512;  // Batch size for receiving updates

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
    global_weights -= LEARNING_RATE * total_gradients;

    //std::cout << "[DEBUG] Updated global weights (first 10 values): "
      //        << global_weights.head(10).transpose() << std::endl;

    // **Reset total_gradients after applying update**
    total_gradients.setZero();
}

// **Handles communication with a client**
void handle_client(tcp::socket socket) {
    try {
       // std::cout << "[DEBUG] New client connected." << std::endl;

        while (true) {
            int batch_size = 0, vector_size = 0;

            // **Read batch size and vector size from client**
            boost::asio::read(socket, boost::asio::buffer(&batch_size, sizeof(int)));
            boost::asio::read(socket, boost::asio::buffer(&vector_size, sizeof(int)));

            if (vector_size <= 0 || vector_size > 1e7) {
                std::cerr << "[ERROR] Invalid vector size received: " << vector_size << std::endl;
                return;
            }

            //std::cout << "[DEBUG] Receiving batch of size: " << batch_size << ", Vector size: " << vector_size << std::endl;

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

          //  std::cout << "[DEBUG] Sent updated global model to client after batch." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in handle_client: " << e.what() << std::endl;
    }
}

void handle_client_pinned(tcp::socket socket, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "[ERROR] Failed to set thread affinity to core " << core_id << ": " << strerror(errno) << std::endl;
    } else {
       // std::cout << "[INFO] Thread pinned to core " << core_id << std::endl;
    }

    handle_client(std::move(socket));
}


// **Main function to run the server**
int main() {
    
    try {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 12344));

        //std::cout << "[DEBUG] Server started..." << std::endl;
        int core_id = -1;  // You can increment this in a round-robin fashion
        while (true) {
            tcp::socket socket(io_context);
            acceptor.accept(socket);
           // std::cout << "[DEBUG] Client connected." << std::endl;

            //std::thread(handle_client, std::move(socket)).detach();
            core_id = (core_id + 1) % 26;  // round-robin core assignment
            std::thread([core_id](tcp::socket s) {
                handle_client_pinned(std::move(s), core_id);
            }, std::move(socket)).detach();
        
           
        } 
        

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in server: " << e.what() << std::endl;
    }

    return 0;
}
