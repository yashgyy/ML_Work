#!/bin/bash
# Function to kill processes on port 12344
kill_port_12344() {
    lsof -i:12344 -t | xargs -r kill -9 2>/dev/null
}
# Function to wait for user confirmation
wait_for_confirmation() {
    local app_name=$1
    echo "========================================="
    echo "Ready to profile: $app_name (SERVER)"
    echo "Coordinate with client script operator!"
    read -p "Press Enter when ready to start profiling $app_name server..."
}
# Function to run profiling for an application
run_profiling() {
    local app_path=$1
    local output_path=$2
    local app_name=$3
    
    echo "Starting server profiling for $app_name..."
    kill_port_12344
    
    # Create output directory if it doesn't exist
    local output_dir=$(dirname "$output_path")
    mkdir -p "$output_dir"
    echo "Created output directory: $output_dir"
    
    # Start server with profiling using perf
    echo "Launching server with profiler..."
    perf stat --timeout 300000 -e cycles,instructions -o "$output_path" -x , "$app_path" 
    
    echo "Server profiling completed for $app_name"
    echo "Output saved to: $output_path"
    echo ""
}
# Main execution
echo "Perf Server Profiling Script (Iterative Mode)"
echo "Make sure you have appropriate permissions for perf!"
echo ""
# Define applications to profile
declare -A applications=(
    ["Naive_Bayes"]="../Naive_Bayes/server ../Naive_Bayes/IS_AMDC_ipc_performance_nb.csv"
    ["Logistic_Regression"]="../Logistic_Regression/server ../Logistic_Regression/IS_AMDC_ipc_performance_lr.csv"
    ["Linear_Regression"]="../Linear_Regression/server ../Linear_Regression/IS_AMDC_ipc_performance_linear.csv"
    ["KernelSVM"]="../KernelSVM/server ../KernelSVM/IS_AMDC_ipc_performance_ksvm.csv"
    ["LSVM"]="../LSVM/server ../LSVM/IS_AMDC_ipc_performance_lsvm.csv"
    ["KMeans"]="../KMeans/server ../KMeans/IS_AMDC_ipc_performance_kmeans.csv"
    ["Adaboost"]="../Adaboost/server ../Adaboost/IS_AMDC_ipc_performance_adaboost.csv"
    ["RF"]="../RF/server ../RF/IS_AMDC_ipc_performance_rf.csv"
)
# Process each application
for app_name in "KernelSVM" "Adaboost" "KMeans" "Naive_Bayes" "Logistic_Regression" "Linear_Regression" "LSVM" "RF"; do
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
                echo "Completed profiling for $app_name server"
                read -p "Ready for the next application? Press Enter to continue..."
                echo ""
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