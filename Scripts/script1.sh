#!/bin/bash

#AMD=/opt/AMDuProf_5.0-1479/bin/AMDuProfCLI

# Function to kill any process using port 8080
kill_port_8080() {
    lsof -i:8080 -t | xargs -r kill -9
}

# Create necessary directories
mkdir -p ~/Work/Profile/KernelSVM/server
mkdir -p ~/Work/Profile/KernelSVM/client

# List of configurations with start delay
configs_with_delay=("memory" "cpi" "inst_access" "branch" "data_access" "ibs" "tbp" "hotspots" "assess" "assess_ext")

# List of configurations without start delay
configs_no_delay=("threading" "overview")

kill_port_8080

# # Loop through configurations with start delay
# for config in "${configs_with_delay[@]}"; do
#     echo "RUN $config"
#     echo "Collecting $config profile for server and client with start delay..."

#     # Run profiling with start delay for server and client
#     $AMD profile --detail  --show-all-cachelines --config "$config" --duration 20 --start-delay 5 --output-dir "~/Work/Profile/KernelSVM/server/$config" "Work/Profiling_Fed/ML/KernelSVM/server" &
#     $AMD profile --detail  --show-all-cachelines --config "$config" --duration 20 --start-delay 5 --output-dir "~/Work/Profile/KernelSVM/client/$config" "Work/Profiling_Fed/ML/KernelSVM/client"

#     # folder_name=$(find "/home/yash/Work/Profile/KernelSVM/server/$config" -type d -name "AMDu*" -print -quit)

#     # $AMD report --detail --show-all-cachelines -i "$folder_name" --report-output "$folder_name"

#     # folder_name=$(find "/home/yash/Work/Profile/KernelSVM/client/$config" -type d -name "AMDu*" -print -quit)

#     # $AMD report --detail --show-all-cachelines -i "$folder_name" --report-output "$folder_name"

#     # Kill any processes on port 8080 after each profiling command
#     kill_port_8080
#     echo -e "------------ \n"
# done

#Wait for 10 seconds before starting the next loop
echo "Waiting for 10 seconds before starting configurations without start delay..."
sleep 10
kill_port_8080
# Loop through configurations without start delay
for config in "${configs_no_delay[@]}"; do
    echo "RUN $config"
    echo "Collecting $config profile for server and client without start delay..."

    # Run profiling without start delay for server and client
    $AMD collect  --config "$config" --duration 60 --output-dir "~/Work/Profile/KernelSVM/server/$config" "Work/Profiling_Fed/ML/KernelSVM/server" &
    $AMD collect  --config "$config" --duration 60 --output-dir "~/Work/Profile/KernelSVM/client/$config" "Work/Profiling_Fed/ML/KernelSVM/client"

    # Kill any processes on port 8080 after each profiling command
    kill_port_8080

    echo -e "------------ \n"
done
kill_port_8080