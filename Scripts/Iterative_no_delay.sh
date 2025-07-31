#!/bin/bash

# Function to kill processes on port 12344
kill_port_12344() {
    lsof -i:12344 -t | xargs -r kill -9 2>/dev/null
}

# Function to wait for user confirmation (first app only)
wait_for_confirmation() {
    local app_name=$1
    echo "========================================="
    echo "Ready to profile: $app_name (SERVER)"
    echo "Coordinate with client script operator!"
    read -p "Press Enter when ready to start profiling $app_name server..."
}

# Function to wait with countdown for automatic start
wait_with_countdown() {
    local app_name=$1
    echo "========================================="
    echo "Next profiling: $app_name (SERVER)"
    echo "Starting automatically in 60 seconds..."
    
    for i in {60..1}; do
        printf "\rStarting $app_name server in %2d seconds... (Press Ctrl+C to abort)" $i
        sleep 1
    done
    printf "\rStarting $app_name server now!                                        \n"
}

# Function to run profiling for an application
run_profiling() {
    local app_path=$1
    local output_path=$2
    local app_name=$3
    
    echo "Starting server profiling for $app_name..."
    kill_port_12344
    
    AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
    
    # Start server with profiling
    echo "Launching server with profiler..."
    $AMD1 -m ipc,fp,l1,l2,memory,l3 -d 300 -o "$output_path" -- "$app_path"
    
    echo "Server profiling completed for $app_name"
    echo "Output saved to: $output_path"
    echo ""
}

# Main execution
echo "AMD uProf Server Profiling Script (Iterative Mode)"
echo "Make sure to run 'sudo modprobe msr' first!"
echo ""

# Define applications to profile
declare -A applications=(
    ["Naive_Bayes"]="../Naive_Bayes/server ../Naive_Bayes/Server/performance.csv"
    ["Logistic_Regression"]="../Logistic_Regression/server ../Logistic_Regression/Server/performance.csv"
    ["Linear_Regression"]="../Linear_Regression/server ../Linear_Regression/Server/performance.csv"
    ["KernelSVM"]="../KernelSVM/server ../KernelSVM/Server/performance.csv"
    ["LSVM"]="../LSVM/server ../LSVM/Server/performance.csv"
    ["KMeans"]="../KMeans/server ../KMeans/Server/performance.csv"
    ["Adaboost"]="../Adaboost/server ../Adaboost/Server/performance.csv"
    ["RF"]="../RF/server ../RF/Server/performance.csv"
)

# Process each application
first_app=true
for app_name in "Naive_Bayes" "Logistic_Regression" "Linear_Regression" "KernelSVM" "LSVM" "KMeans" "Adaboost" "RF"; do
    if [[ -n "${applications[$app_name]}" ]]; then
        # Parse application path and output path
        app_info=(${applications[$app_name]})
        app_path=${app_info[0]}
        output_path=${app_info[1]}
        
        # Check if application exists
        if [[ -f "$app_path" ]]; then
            # Use manual confirmation for first app, automatic countdown for rest
            if [[ "$first_app" == true ]]; then
                wait_for_confirmation "$app_name"
                first_app=false
            else
                wait_with_countdown "$app_name"
            fi
            
            run_profiling "$app_path" "$output_path" "$app_name"
            
            # Add 1-minute delay after profiling (except for last application)
            if [[ "$app_name" != "RF" ]]; then
                echo "Profiling completed for $app_name server"
                echo "Waiting 1 minute before next application..."
                echo ""
                sleep 60
            fi
        else
            echo "Warning: Server application not found: $app_path"
            echo "Skipping $app_name..."
            echo ""
        fi
    fi
done

echo "All server profiling completed!"
echo "Remember to collect the performance.csv files from each application's Server directory."