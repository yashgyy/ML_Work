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
   
    # Create result directory name based on app name
    result_dir="${app_name}_vtune_results"
    
    # Launch client instances in background first
    echo "Launching 26 client instances for $app_name..."
    for i in {1..25}; do  
        $app_path &
    done
   
    # Start VTune profiling (this will profile the main client process and collect data for 300 seconds)
    echo "Starting VTune uarch-exploration collection for $app_name..."
    vtune -collect uarch-exploration -d 300 -data-limit=3000 -knob collect-memory-bandwidth=true -result-dir $result_dir $app_path &
    
    # Store the VTune PID to wait for it later
    vtune_pid=$!
    
    echo "VTune profiling started for $app_name (PID: $vtune_pid)"
    echo "Profiling will run for 300 seconds..."
   
    # Wait for VTune profiling to complete
    wait $vtune_pid
    
    echo "VTune collection completed for $app_name"
    
    # Generate CSV report from VTune results
    echo "Generating CSV report for $app_name..."
    vtune -report hw-events -report-knob show-issues=false -format=csv -csv-delimiter=comma -report-output="$output_path" -group-by process-id -result-dir $result_dir
    
    # Clean up any remaining client processes
    kill_port_12344
   
    echo "Profiling and report generation completed for $app_name"
    echo "VTune results saved to: $result_dir"
    echo "CSV report saved to: $output_path"
    echo ""
}

# Main execution
echo "Intel VTune Client Profiling Script (Iterative Mode)"
echo "Make sure VTune is properly installed and configured!"
echo ""

# Define applications to profile
declare -A applications=(
    ["Naive_Bayes"]="../Naive_Bayes/client.exe ../Naive_Bayes/IC_IS_uarch_nb_main_bandwidth.csv"
    ["Logistic_Regression"]="../Logistic_Regression/client.exe ../Logistic_Regression/IC_IS_uarch_lr_main_bandwidth.csv"
    ["Linear_Regression"]="../Linear_Regression/client.exe ../Linear_Regression/IC_IS_uarch_linear_main_bandwidth.csv"
    ["KernelSVM"]="../KernelSVM/client.exe ../KernelSVM/IC_IS_uarch_ksvm_main_bandwidth.csv"
    ["LSVM"]="../LSVM/client.exe ../LSVM/IC_IS_uarch_lsvm_main_bandwidth.csv"
    ["KMeans"]="../KMeans/client.exe ../KMeans/IC_IS_uarch_kmeans_main_bandwidth.csv"
    ["Adaboost"]="../Adaboost/client.exe ../Adaboost/IC_IS_uarch_adaboost_main_bandwidth.csv"
    ["RF"]="../RF/client.exe ../RF/IC_IS_uarch_rf_main_bandwidth.csv"
)

# Process each application
for app_name in "KMeans" "Naive_Bayes" "Logistic_Regression" "Linear_Regression" "KernelSVM" "LSVM" "Adaboost" "RF"; do
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
echo "VTune result directories and CSV reports have been generated for each application."
echo "CSV files are saved in their respective application directories."