#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Function to load features and labels from a CSV file (using double precision)
void load_data(const std::string& filename, 
               std::vector<std::vector<double>>& features, 
               std::vector<double>& labels) {  // Target is now double
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open file: " << filename << std::endl;
        return;
    }

    // Read the header line and skip it
    std::getline(file, line);

    int expected_columns = 31;  // 30 features + 1 target

    // Read the data lines
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> feature_row;
        double target;

        std::vector<std::string> row_values;
        while (std::getline(ss, value, ',')) {
            row_values.push_back(value);
        }

        // Ensure the row has the correct number of columns
        if (row_values.size() != expected_columns) {
            std::cerr << "[WARNING] Skipping row with incorrect column count: " << row_values.size() << std::endl;
            continue;
        }

        // Extract features (first 30 columns)
        for (int i = 0; i < 30; ++i) {
            feature_row.push_back(std::stod(row_values[i]));  // Convert all features to double
        }

        // Extract target (last column)
        target = std::stod(row_values.back());  // Convert target to double

        // Add to dataset containers
        features.push_back(feature_row);
        labels.push_back(target);
    }

    file.close();
    std::cout << "[INFO] Loaded " << features.size() 
              << " samples with " << features[0].size() 
              << " features each from " << filename << "." << std::endl;
}
