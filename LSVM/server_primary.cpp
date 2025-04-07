// Refactored Federated Linear SVM Server (based on K_Server structure)
#include <iostream>
#include <vector>
#include <thread>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include <mutex>
#include <numeric>
#include "data_loader.cpp"
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

using namespace Eigen;
using boost::asio::ip::tcp;

std::mutex model_mutex;
VectorXd global_weights;
VectorXd total_gradients;
std::vector<int> client_data_sizes;
int client_count = 0;

const int BATCH_SIZE = 512;

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

   // std::cout << "[DEBUG] Updated global weights: " << global_weights.transpose() << std::endl;
    total_gradients.setZero();
}

void handle_client(tcp::socket socket) {
    try {
        //std::cout << "[DEBUG] Client connected." << std::endl;

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
         //   std::cout << "[DEBUG] Sent global model to client." << std::endl;
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


int main() {
    try {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 12344));

       // std::cout << "[INFO] Server started. Waiting for clients...\n";
       int core_id = -1;  // You can increment this in a round-robin fashion
        while (true) {
            tcp::socket socket(io_context);
            acceptor.accept(socket);
            core_id = (core_id + 1) % 26;  // round-robin core assignment
            //std::thread(handle_client, std::move(socket)).detach();
            std::thread([core_id](tcp::socket s) {
                handle_client_pinned(std::move(s), core_id);
            }, std::move(socket)).detach();
        
            
        }        


    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in server: " << e.what() << std::endl;
    }
    return 0;
}
