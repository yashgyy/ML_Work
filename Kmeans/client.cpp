
#include <iostream>
#include <vector>
#include <boost/asio.hpp>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>

using namespace Eigen;
using boost::asio::ip::tcp;
// Function to simulate or load local data
MatrixXd load_local_data() {
    MatrixXd data(4, 2); // 4 samples, 2 features
    data << 1.0, 2.0,
            2.0, 1.0,
            3.0, 4.0,
            4.0, 3.0;
    return data;
}

// Function to randomly select k samples from the data to initialize centroids
MatrixXd initialize_centroids(const MatrixXd& data, int k) {
    std::vector<int> indices = {0, 1, 2, 3};  // Indices of the data rows
    
    // Seed the random generator with a unique value (time + process ID)
    std::srand(std::time(0) + getpid());  // Unique seed for each client instance
    std::random_shuffle(indices.begin(), indices.end());  // Shuffle the indices

    MatrixXd centroids(k, data.cols());
    for (int i = 0; i < k; ++i) {
        centroids.row(i) = data.row(indices[i]);  // Select random rows for centroids
    }

    return centroids;
}

// Function to perform K-means clustering
MatrixXd perform_kmeans(const MatrixXd& data, MatrixXd centroids, int max_iters = 10) {
    int n_samples = data.rows();
    int k = centroids.rows();
    VectorXi labels(n_samples);
    
    for (int iter = 0; iter < max_iters; ++iter) {
        // Assignment step: Assign each point to the nearest centroid
        for (int i = 0; i < n_samples; ++i) {
            RowVectorXd point = data.row(i);
            VectorXd distances = (centroids.rowwise() - point).rowwise().squaredNorm();
            labels(i) = distances.minCoeff();
        }

        // Update step: Update centroids
        for (int j = 0; j < k; ++j) {
            std::vector<RowVectorXd> points_in_cluster;

            for (int i = 0; i < n_samples; ++i) {
                if (labels(i) == j) {
                    points_in_cluster.push_back(data.row(i));
                }
            }

            if (!points_in_cluster.empty()) {
                MatrixXd points_matrix(points_in_cluster.size(), data.cols());

                for (int i = 0; i < points_in_cluster.size(); ++i) {
                    points_matrix.row(i) = points_in_cluster[i];
                }

                centroids.row(j) = points_matrix.colwise().mean();
            }
        }
    }

    return centroids;
}

int main() {
    try {
        boost::asio::io_context io_context;
        tcp::socket socket(io_context);
        socket.connect(tcp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 8080));

        // Load local data and initialize centroids with random variation
        MatrixXd local_data = load_local_data();
        MatrixXd local_centroids = initialize_centroids(local_data, 2);

        // Perform local K-means clustering
        MatrixXd updated_centroids = perform_kmeans(local_data, local_centroids);

        // Send the size of the centroid matrix first
        int rows = updated_centroids.rows();
        int cols = updated_centroids.cols();
        boost::asio::write(socket, boost::asio::buffer(&rows, sizeof(int)));
        boost::asio::write(socket, boost::asio::buffer(&cols, sizeof(int)));

        // Send local centroids to server
        boost::asio::write(socket, boost::asio::buffer(updated_centroids.data(), rows * cols * sizeof(double)));

        // Receive updated global centroids from the server
        MatrixXd global_centroids(rows, cols);
        boost::asio::read(socket, boost::asio::buffer(global_centroids.data(), rows * cols * sizeof(double)));

        std::cout << "Updated global centroids received from server:" << global_centroids << std::endl;

    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
