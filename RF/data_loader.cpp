#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Function to load features and labels from the CSV file
void load_data(const std::string& filename, 
    std::vector<std::vector<float>>& features, 
    std::vector<int>& labels) {
std::ifstream file(filename);
std::string line;
std::cout<<"File Opened"<<std::endl;

if (!file.is_open()) {
std::cerr << "[ERROR] Could not open file: " << filename << std::endl;
return;
}

std::getline(file, line); // Skip header

int row_count = 0;


while (std::getline(file, line)) {
std::stringstream ss(line);
std::string value;
std::vector<float> feature_row;
int label;

int column_index = 0;
while (std::getline(ss, value, ',')) {
 if (column_index == 1) {
     label = std::stoi(value);
 } else if (column_index >= 2) {
     feature_row.push_back(std::stof(value));
 }
 column_index++;
}

features.push_back(feature_row);
labels.push_back(label);
row_count++;
}

file.close();
std::cout << "[INFO] Loaded " << features.size() 
   << " samples with " << features[0].size() 
   << " features each from " << filename << "." << std::endl;
}