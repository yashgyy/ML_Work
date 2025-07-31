#!/bin/bash

# Function to kill processes on port 12344
kill_port_12344() {
    lsof -i:12344 -t | xargs -r kill -9 2>/dev/null
}

# Function to wait for user confirmation (first app only)
wait_for_confirmation() {
    local app_name=$1
    echo "========================================="
    echo "Ready to profile: $app_name"
    echo "Make sure the server script is ready!"
    read -p "Press Enter when ready to start profiling $app_name..."
}

# Function to wait with countdown for automatic start
wait_with_countdown() {
    local app_name=$1
    echo "========================================="
    echo "Next profiling: $app_name"
    echo "Starting automatically in 60 seconds..."
    
    for i in {60..1}; do
        printf "\rStarting $app_name in %2d seconds... (Press Ctrl+C to abort)" $i
        sleep 1
    done
    printf "\rStarting $app_name now!                                        \n"
}

# Function to run profiling for an application
run_profiling() {
    local app_path=$1
    local output_path=$2
    local app_name=$3
    
    echo "Starting profiling for $app_name..."
    kill_port_12344
    
    AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
    
    # Start profiler in background
    $AMD1 -m ipc,fp,l1,l2,l3 -d 300 -o "$output_path" -- "$app_path" &
    
    # Give profiler time to start
    sleep 2
    
    # Launch client instances
    for i in {1..26}; do  
        $app_path &
    done
    
    echo "Started 26 client instances for $app_name"
    echo "Profiling will run for 300 seconds..."
    
    # Wait for profiling to complete
    wait
    
    echo "Profiling completed for $app_name"
    echo "Output saved to: $output_path"
    echo ""
}

# Main execution
echo "AMD uProf Client Profiling Script (Iterative Mode)"
echo "Make sure to run 'sudo modprobe msr' first!"
echo ""

# Define applications to profile
declare -A applications=(
    ["Naive_Bayes"]="../Naive_Bayes/client ../Naive_Bayes/Client/performance.csv"
    ["Logistic_Regression"]="../Logistic_Regression/client ../Logistic_Regression/Client/performance.csv"
    ["Linear_Regression"]="../Linear_Regression/client ../Linear_Regression/Client/performance.csv"
    ["KernelSVM"]="../KernelSVM/client ../KernelSVM/Client/performance.csv"
    ["LSVM"]="../LSVM/client ../LSVM/Client/performance.csv"
    ["KMeans"]="../KMeans/client ../KMeans/Client/performance.csv"
    ["Adaboost"]="../Adaboost/client ../Adaboost/Client/performance.csv"
    ["RF"]="../RF/client ../RF/Client/performance.csv"
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
                echo "Profiling completed for $app_name"
                echo "Waiting 1 minute before next application..."
                echo ""
                sleep 60
            fi
        else
            echo "Warning: Application not found: $app_path"
            echo "Skipping $app_name..."
            echo ""
        fi
    fi
done

echo "All profiling completed!"
echo "Remember to collect the performance.csv files from each application's Client directory."