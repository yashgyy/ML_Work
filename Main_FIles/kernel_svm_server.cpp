#include <iostream>
#include <vector>
#include <thread>
#include <boost/asio.hpp>
#include <mutex>
#include <Eigen/Dense>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <numeric>

using namespace Eigen;
using boost::asio::ip::tcp;

std::mutex model_mutex;
VectorXd global_weights;
VectorXd total_gradients;
std::vector<int> client_data_sizes;
int client_count = 0;

const int BATCH_SIZE = 30;
const double LEARNING_RATE = 0.5;

void apply_kernel_gradient_update(const VectorXd& gradient_vector, int batch_size) {
    std::lock_guard<std::mutex> lock(model_mutex);
    //std::cout << "[DEBUG] Received gradient update from client." << std::endl;
    if (global_weights.size() == 0) {
        global_weights = VectorXd::Zero(gradient_vector.size());
        total_gradients = VectorXd::Zero(gradient_vector.size());
    }

    total_gradients += gradient_vector * batch_size;
    client_data_sizes.push_back(batch_size);
    client_count++;

    int total_points = std::accumulate(client_data_sizes.begin(), client_data_sizes.end(), 0);
    global_weights -= LEARNING_RATE * (total_gradients / total_points);

    total_gradients.setZero();
}

void handle_client(tcp::socket socket) {
    try {
        while (true) {
            int batch_size = 0;
            int vector_size = 0;
            //std::cout << "[DEBUG] Waiting for batch size and vector size from client." << std::endl;
            boost::asio::read(socket, boost::asio::buffer(&batch_size, sizeof(int)));
            boost::asio::read(socket, boost::asio::buffer(&vector_size, sizeof(int)));
            //std::cout << "[DEBUG] Received batch size: " << batch_size << ", vector size: " << vector_size << std::endl;
            if (vector_size <= 0 || vector_size > 1e9) {
                std::cerr << "[ERROR] Invalid vector size: " << vector_size << std::endl;
                return;
            }

            VectorXd gradient_vector = VectorXd::Zero(vector_size);
            int received = 0;

            while (received < vector_size) {
                int chunk_size = std::min(BATCH_SIZE, vector_size - received);
                boost::asio::read(socket, boost::asio::buffer(gradient_vector.data() + received, chunk_size * sizeof(double)));
                received += chunk_size;
            }

            apply_kernel_gradient_update(gradient_vector, batch_size);

            boost::asio::write(socket, boost::asio::buffer(global_weights.data(), global_weights.size() * sizeof(double)));
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
    }
    //std::cout << "[INFO] Thread pinned to core " << core_id << std::endl;

    handle_client(std::move(socket));
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 12344));
        int core_id = -1;

        while (true) {
            tcp::socket socket(io_context);
            acceptor.accept(socket);
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