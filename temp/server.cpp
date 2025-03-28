#include <iostream>
#include <vector>
#include <thread>
#include <boost/asio.hpp>
#include <mutex>
#include <Eigen/Dense>

using namespace Eigen;
using boost::asio::ip::tcp;

std::mutex model_mutex;
VectorXd global_weights;
VectorXd total_weights;
int client_count = 0;

const int BATCH_SIZE = 30;

void aggregate_model(const VectorXd& local_update) {
    std::lock_guard<std::mutex> lock(model_mutex);
    if (total_weights.size() == 0) total_weights = VectorXd::Zero(local_update.size());
    total_weights += local_update;
    client_count++;
    global_weights = total_weights / client_count;
}

void handle_client(tcp::socket socket) {
    try {

        std::cout<<"Client Connected"<<std::endl;
        
        //boost::asio::read(socket, boost::asio::buffer(&vector_size, sizeof(int)));

        while (true) {
            int batch_size = 0;
            int vector_size = 0;
            //std::cout<<"Before Read"<<std::endl;
            boost::asio::read(socket, boost::asio::buffer(&batch_size, sizeof(int)));
            boost::asio::read(socket, boost::asio::buffer(&vector_size, sizeof(int)));
            //std::cout<<"After Read"<<std::endl;
            if (vector_size <= 0 || vector_size > 1e9) {
                std::cerr << "[ERROR] Invalid vector size: " << vector_size << std::endl;
                return;
            }

            VectorXd local_update = VectorXd::Zero(vector_size);
            int received = 0;

            while (received < vector_size) {
                int chunk_size = std::min(BATCH_SIZE, vector_size - received);
                boost::asio::read(socket, boost::asio::buffer(local_update.data() + received, chunk_size * sizeof(double)));
                received += chunk_size;
            }

            aggregate_model(local_update);

            boost::asio::write(socket, boost::asio::buffer(global_weights.data(), global_weights.size() * sizeof(double)));
            std::cout<<"Send and Recieved"<<std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in handle_client: " << e.what() << std::endl;
    }
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 8080));
        std::cout << "[INFO] Server started. Waiting for clients..." << std::endl;

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
