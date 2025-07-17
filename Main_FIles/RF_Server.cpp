// Federated Random Forest Server (Batch-wise Epoch-based Aggregation)
#include <iostream>
#include <vector>
#include <thread>
#include <boost/asio.hpp>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

using boost::asio::ip::tcp;

struct DecisionTree {
    int feature_index;
    float threshold;
    int class_label;
};

std::mutex forest_mutex;
std::vector<DecisionTree> global_forest;

std::vector<DecisionTree> deserialize_trees(const std::vector<double>& serialized, int count) {
    std::vector<DecisionTree> trees;
    for (int i = 0; i < count; ++i) {
        trees.push_back({static_cast<int>(serialized[i * 3]),
                         static_cast<float>(serialized[i * 3 + 1]),
                         static_cast<int>(serialized[i * 3 + 2])});
    }
    return trees;
}

std::vector<double> serialize_trees(const std::vector<DecisionTree>& trees) {
    std::vector<double> serialized;
    for (const auto& tree : trees) {
        serialized.push_back(tree.feature_index);
        serialized.push_back(tree.threshold);
        serialized.push_back(tree.class_label);
    }
    return serialized;
}

void handle_client(tcp::socket socket) {
    try {
        //std::cout << "[INFO] New client connected.\n";

        while (true) {
            int num_trees = 0, vec_size = 0;
            boost::system::error_code ec;
            boost::asio::read(socket, boost::asio::buffer(&num_trees, sizeof(int)), ec);
            if (ec) break; // Client closed connection

            boost::asio::read(socket, boost::asio::buffer(&vec_size, sizeof(int)));
            std::vector<double> serialized(vec_size);
            boost::asio::read(socket, boost::asio::buffer(serialized.data(), vec_size * sizeof(double)));

            std::vector<DecisionTree> trees = deserialize_trees(serialized, num_trees);
            {
                std::lock_guard<std::mutex> lock(forest_mutex);
                global_forest.insert(global_forest.end(), trees.begin(), trees.end());
            }

            std::vector<double> global_serialized;
            {
                std::lock_guard<std::mutex> lock(forest_mutex);
                global_serialized = serialize_trees(global_forest);
            }

            int global_vec_size = global_serialized.size();
            boost::asio::write(socket, boost::asio::buffer(&global_vec_size, sizeof(int)));
            boost::asio::write(socket, boost::asio::buffer(global_serialized.data(), global_vec_size * sizeof(double)));
        }

        //std::cout << "[INFO] Client disconnected.\n";

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in server handler: " << e.what() << std::endl;
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
        //std::cout << "[INFO] Federated RF Server running on port 12344...\n";
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
        std::cerr << "[ERROR] Exception in main: " << e.what() << std::endl;
    }
    return 0;
}