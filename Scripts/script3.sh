
# Function to kill any process using port 8080
kill_port_8080() {
    lsof -i:8080 -t | xargs -r kill -9
}


# Create necessary directories
mkdir -p ~/Work/Profile/KernelSVM/server
mkdir -p ~/Work/Profile/KernelSVM/client


kill_port_8080

AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm

# $AMD1 -m ipc,fp,l1,l2,swpfdc,hwpfdc -d 360 -o ~/Work/Profile/KernelSVM/server/perfomance.csv -- "Work/Profiling_Fed/ML/KernelSVM/server" &
# $AMD1 -m ipc,fp,l1,l2,swpfdc,hwpfdc -d 360 -o ~/Work/Profile/KernelSVM/client/perfomance.csv -- "Work/Profiling_Fed/ML/KernelSVM/client"

$AMD1 -m l3 -d 360 -o ~/Work/Profile/KernelSVM/server/perfomance.csv -- "Work/Profiling_Fed/ML/KernelSVM/server" &
$AMD1 -m l3 -d 360 -o ~/Work/Profile/KernelSVM/client/perfomance.csv -- "Work/Profiling_Fed/ML/KernelSVM/client"
