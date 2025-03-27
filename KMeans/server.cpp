// Refactored Federated KMeans Server (Based on K_Server structure)
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <boost/asio.hpp>
#include <Eigen/Dense>

using namespace Eigen;
using boost::asio::ip::tcp;

std::mutex mutex_lock;
MatrixXd global_centroids;
int clients_count = 0;

void receive_matrix(tcp::socket& socket, MatrixXd& matrix) {
    int rows, cols;
    boost::asio::read(socket, boost::asio::buffer(&rows, sizeof(int)));
    boost::asio::read(socket, boost::asio::buffer(&cols, sizeof(int)));
    matrix.resize(rows, cols);
    boost::asio::read(socket, boost::asio::buffer(matrix.data(), rows * cols * sizeof(double)));
}

void send_matrix(tcp::socket& socket, const MatrixXd& matrix) {
    int rows = matrix.rows(), cols = matrix.cols();
    boost::asio::write(socket, boost::asio::buffer(&rows, sizeof(int)));
    boost::asio::write(socket, boost::asio::buffer(&cols, sizeof(int)));
    boost::asio::write(socket, boost::asio::buffer(matrix.data(), rows * cols * sizeof(double)));
}

void handle_client(tcp::socket socket) {
    try {
        for (int round = 0; round < 100; ++round) {
            MatrixXd local_centroids;
            receive_matrix(socket, local_centroids);

            std::lock_guard<std::mutex> lock(mutex_lock);
            if (clients_count == 0) {
                global_centroids = local_centroids;
            } else {
                global_centroids += local_centroids;
            }
            clients_count++;

            MatrixXd avg_centroids = global_centroids / clients_count;
            send_matrix(socket, avg_centroids);

            std::cout << "[DEBUG] Round " << round + 1 << ": Updated global centroids:\n" << avg_centroids << std::endl;
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