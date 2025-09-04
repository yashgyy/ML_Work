#!/bin/bash

# Function to kill processes on port 12344
kill_port_12344() {
    lsof -i:12344 -t | xargs -r kill -9 2>/dev/null
}

# Function to wait for user confirmation
wait_for_confirmation() {
    local app_name=$1
    echo "========================================="
    echo "Ready to profile: $app_name"
    echo "Make sure the server script is ready!"
    read -p "Press Enter when ready to start profiling $app_name..."
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
    perf stat --timeout 300000 -e L1-icache-load-misses,icache_64b.iftag_hit,icache_64b.iftag_miss,L1-dcache-load-misses,L1-dcache-loads,l2_rqsts.miss,l2_rqsts.references,LLC-load-misses,LLC-loads,l2_rqsts.all_code_rd -o "$output_path" -x , "$app_path" &
    
    # Give profiler time to start
    sleep 1
    
    # Launch client instances
    for i in {1..25}; do  
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
    ["Naive_Bayes"]="../Naive_Bayes/client ../Naive_Bayes/IC_AMDS_micro_performance.csv"
    ["Logistic_Regression"]="../Logistic_Regression/client ../Logistic_Regression/IC_AMDS_micro_performance.csv"
    ["Linear_Regression"]="../Linear_Regression/client ../Linear_Regression/IC_AMDS_micro_performance.csv"
    ["KernelSVM"]="../KernelSVM/client ../KernelSVM/IC_AMDS_micro_performance.csv"
    ["LSVM"]="../LSVM/client ../LSVM/IC_AMDS_micro_performance.csv"
    ["KMeans"]="../KMeans/client ../KMeans/IC_AMDS_micro_performance.csv"
    ["Adaboost"]="../Adaboost/client ../Adaboost/IC_AMDS_micro_performance.csv"
    ["RF"]="../RF/client ../RF/IC_AMDS_micro_performance.csv"
)

# Process each application
for app_name in "Naive_Bayes" "Logistic_Regression" "Linear_Regression" "KernelSVM" "LSVM" "KMeans" "Adaboost" "RF"; do
    if [[ -n "${applications[$app_name]}" ]]; then
        # Parse application path and output path
        app_info=(${applications[$app_name]})
        app_path=${app_info[0]}
        output_path=${app_info[1]}
        
        # Check if application exists
        if [[ -f "$app_path" ]]; then
            wait_for_confirmation "$app_name"
            run_profiling "$app_path" "$output_path" "$app_name"
            
            # Ask if ready for next application
            if [[ "$app_name" != "RF" ]]; then  # Don't ask after the last application
                echo "Completed profiling for $app_name"
                read -p "Ready for the next application? Press Enter to continue..."
                echo ""
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