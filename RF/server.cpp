#include <iostream>
#include <vector>
#include <thread>
#include <boost/asio.hpp>
#include <mutex>
#include <Eigen/Dense>
#include <map>

using namespace Eigen;
using boost::asio::ip::tcp;

struct TreeNode {
    int feature_index = -1;
    double threshold = 0.0;
    double prediction = 0.0;
    TreeNode* left = nullptr;
    TreeNode* right = nullptr;
};

std::mutex model_mutex;
std::vector<std::vector<TreeNode*>> aggregated_forests;

TreeNode* deserialize_tree_node(const std::vector<double>& serialized_tree, int& index);
void perform_predictions();
void aggregate_forests(const std::vector<TreeNode*>& forest);

void handle_client(tcp::socket socket) {
    try {
        int forest_size;
        boost::asio::read(socket, boost::asio::buffer(&forest_size, sizeof(int)));

        std::vector<TreeNode*> local_forest(forest_size);
        for (int i = 0; i < forest_size; ++i) {
            int tree_size;
            boost::asio::read(socket, boost::asio::buffer(&tree_size, sizeof(int)));

            std::vector<double> serialized_tree(tree_size);
            boost::asio::read(socket, boost::asio::buffer(serialized_tree.data(), tree_size * sizeof(double)));

            int index = 0;
            local_forest[i] = deserialize_tree_node(serialized_tree, index);
        }

        std::lock_guard<std::mutex> lock(model_mutex);
        aggregated_forests.push_back(local_forest);

        std::cout << "[INFO] Forest from client received and aggregated.\n";

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in handle_client: " << e.what() << std::endl;
    }
}

int main() {
    try {
        boost::asio::io_service io_service;
        tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), 8080));

        std::cout << "[INFO] Server started on port 8080...\n";

        while (true) {
            tcp::socket socket(io_service);
            acceptor.accept(socket);
            std::thread(handle_client, std::move(socket)).detach();

            perform_predictions();
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in main: " << e.what() << std::endl;
    }
    return 0;
}
