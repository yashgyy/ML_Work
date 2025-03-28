// Updated Federated AdaBoost Server with Epoch-compatible Handling
#include <iostream>
#include <vector>
#include <thread>
#include <boost/asio.hpp>
#include <mutex>
#include <Eigen/Dense>
#include <algorithm>

using namespace Eigen;
using boost::asio::ip::tcp;

struct WeakLearner {
    int feature_index;
    double threshold;
    double alpha;
};

std::mutex model_mutex;
std::vector<WeakLearner> aggregated_learners;

std::vector<WeakLearner> deserialize_learners(const std::vector<double>& serialized, int num_learners) {
    std::vector<WeakLearner> learners;
    for (int i = 0; i < num_learners; ++i) {
        WeakLearner learner;
        learner.feature_index = static_cast<int>(serialized[3 * i]);
        learner.threshold = serialized[3 * i + 1];
        learner.alpha = serialized[3 * i + 2];
        learners.push_back(learner);
    }
    return learners;
}

std::vector<double> serialize_learners(const std::vector<WeakLearner>& learners) {
    std::vector<double> serialized;
    for (const auto& learner : learners) {
        serialized.push_back(learner.feature_index);
        serialized.push_back(learner.threshold);
        serialized.push_back(learner.alpha);
    }
    return serialized;
}

double predict_adaboost(const RowVectorXd& sample, const std::vector<WeakLearner>& learners) {
    double prediction = 0.0;
    for (const auto& learner : learners) {
        double stump = (sample(learner.feature_index) <= learner.threshold) ? 1 : -1;
        prediction += learner.alpha * stump;
    }
    return prediction >= 0 ? 1 : -1;
}

void handle_client(tcp::socket socket) {
    try {
        std::cout << "[INFO] Client connected for multiple epochs.\n";

        while (true) {
            int num_learners = 0, vec_size = 0;
            boost::system::error_code ec;

            boost::asio::read(socket, boost::asio::buffer(&num_learners, sizeof(int)), ec);
            if (ec) break; // client closed connection

            boost::asio::read(socket, boost::asio::buffer(&vec_size, sizeof(int)));
            std::vector<double> serialized(vec_size);
            boost::asio::read(socket, boost::asio::buffer(serialized.data(), vec_size * sizeof(double)));

            std::vector<WeakLearner> learners = deserialize_learners(serialized, num_learners);

            {
                std::lock_guard<std::mutex> lock(model_mutex);
                aggregated_learners.insert(aggregated_learners.end(), learners.begin(), learners.end());
            }

            std::cout << "[INFO] Received " << num_learners << " learners from client (epoch loop).\n";

            // Aggregate and return top N learners
            std::vector<WeakLearner> top_learners;
            {
                std::lock_guard<std::mutex> lock(model_mutex);
                top_learners = aggregated_learners;
            }
            std::sort(top_learners.begin(), top_learners.end(), [](const WeakLearner& a, const WeakLearner& b) {
                return a.alpha > b.alpha;
            });

            int N = std::min(20, static_cast<int>(top_learners.size()));
            std::vector<WeakLearner> top_N_learners(top_learners.begin(), top_learners.begin() + N);
            std::vector<double> global_serialized = serialize_learners(top_N_learners);
            int global_vec_size = global_serialized.size();

            boost::asio::write(socket, boost::asio::buffer(&global_vec_size, sizeof(int)));
            boost::asio::write(socket, boost::asio::buffer(global_serialized.data(), global_vec_size * sizeof(double)));
        }

        std::cout << "[INFO] Client disconnected.\n";

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in handle_client: " << e.what() << std::endl;
    }
}

void evaluate_on_dummy_data() {
    if (aggregated_learners.size() >= 5) {
        MatrixXd test_data(4, 2);
        test_data << 1, 2,
                     2, 1,
                     3, 4,
                     4, 3;

        std::cout << "\n[INFO] Predictions on dummy test data:\n";
        for (int i = 0; i < test_data.rows(); ++i) {
            double pred = predict_adaboost(test_data.row(i), aggregated_learners);
            std::cout << "Sample " << i << ": " << pred << std::endl;
        }
    }
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 8080));
        std::cout << "[INFO] Server is running on port 8080...\n";

        while (true) {
            tcp::socket socket(io_context);
            acceptor.accept(socket);
            std::thread(handle_client, std::move(socket)).detach();
            //evaluate_on_dummy_data();
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in server: " << e.what() << std::endl;
    }
    return 0;
}
