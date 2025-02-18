void receive_global_forest(tcp::socket& socket, RandomForest& global_forest) {
    int forest_size;
    boost::asio::read(socket, boost::asio::buffer(&forest_size, sizeof(int)));

    global_forest.trees.resize(forest_size);
    for (int i = 0; i < forest_size; ++i) {
        int tree_size;
        boost::asio::read(socket, boost::asio::buffer(&tree_size, sizeof(int)));

        std::vector<double> serialized_tree(tree_size);
        boost::asio::read(socket, boost::asio::buffer(serialized_tree.data(), tree_size * sizeof(double)));

        int index = 0;
        global_forest.trees[i] = deserialize_tree_node(serialized_tree, index);
    }

    std::cout << "[INFO] Global forest received from the server." << std::endl;
}

void integrate_global_model(const RandomForest& global_forest, RandomForest& local_forest) {
    // Optionally integrate the global forest into the local training process
    local_forest.trees.insert(local_forest.trees.end(), global_forest.trees.begin(), global_forest.trees.end());
}

int main() {
    try {
        boost::asio::io_service io_service;
        tcp::socket socket(io_service);
        tcp::resolver resolver(io_service);
        boost::asio::connect(socket, resolver.resolve({"127.0.0.1", "8080"}));

        MatrixXd data = load_local_data();
        VectorXd labels = load_local_labels();

        RandomForest local_forest(5);
        local_forest.train(data, labels);

        send_forest_to_server(socket, local_forest);

        RandomForest global_forest(0);
        receive_global_forest(socket, global_forest);

        integrate_global_model(global_forest, local_forest);

        std::cout << "[INFO] Local training updated with global forest." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in client: " << e.what() << std::endl;
    }
    return 0;
}
