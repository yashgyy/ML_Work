
#include <iostream>
#include <vector>
#include <boost/asio.hpp>
#include <Eigen/Dense>

using namespace Eigen;
using boost::asio::ip::tcp;

MatrixXd global_centroids_sum;
int num_clients = 0;

// Function to accumulate centroids from clients
void accumulate_centroids(const MatrixXd& local_centroids) {
    if (num_clients == 0) {
        global_centroids_sum = local_centroids;
    } else {
        global_centroids_sum += local_centroids;
    }
    num_clients++;
}

void handle_client(tcp::socket socket) {
    try {
        // Read the size of the centroid matrix from client
        int rows, cols;
        boost::asio::read(socket, boost::asio::buffer(&rows, sizeof(int)));
        boost::asio::read(socket, boost::asio::buffer(&cols, sizeof(int)));

        // Read the centroids from client
        MatrixXd local_centroids(rows, cols);
        boost::asio::read(socket, boost::asio::buffer(local_centroids.data(), rows * cols * sizeof(double)));

        std::cout << "Received local centroids from client:" << local_centroids << std::endl;

        // Accumulate local centroids
        accumulate_centroids(local_centroids);

        // Calculate the average centroids
        MatrixXd global_centroids = global_centroids_sum / num_clients;

        // Send updated global centroids back to client
        boost::asio::write(socket, boost::asio::buffer(global_centroids.data(), rows * cols * sizeof(double)));

    } catch (std::exception& e) {
        std::cerr << "Exception in handle_client: " << e.what() << std::endl;
    }
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 8080));

        std::cout << "Server started. Waiting for clients..." << std::endl;

        while (true) {
            tcp::socket socket(io_context);
            acceptor.accept(socket);
            std::thread(handle_client, std::move(socket)).detach();
        }

    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
