#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

// Function to load features and labels from the CSV file
void load_data(const std::string& filename, 
               std::vector<float>& features, 
               std::vector<int>& labels) {
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open file: " << filename << std::endl;
        return;
    }
    //std::cout<<filename;
    // Read the header line and skip it
    std::getline(file, line);

    // Read the data lines
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> feature_row;
        int label;

        int column_index = 0;
        while (std::getline(ss, value, ',')) {
            if (column_index == 0) {
                // 'target' column (label)
                label = std::stoi(value);
                labels.push_back(label);
            } else if (column_index ==1) {
                // Feature columns (var_0 to var_199)
                features.push_back(std::stof(value));
            }
            column_index++;
        }

        // Add the extracted features and label to the respective containers
        
    }



    file.close();
    std::cout << "[INFO] Loaded " << features.size() 
              << " features each from " << filename << "." << std::endl;
}
