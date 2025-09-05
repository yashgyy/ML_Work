# !/bin/bash
kill_port_12344() {
    lsof -i:12344 -t | xargs -r kill -9
}
kill_port_12344
# AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
# APP="../Naive_Bayes/server"  # Replace with your application name/path
# $AMD1 -m ipc,fp,l1,l2 -d 300 -o ../Naive_Bayes/Server/perfomance.csv -- "../Naive_Bayes/server" 
# echo "Started"
# Define the application you want to run
# Loop to run the application 10 times

# kill_port_12344
# AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
# APP="../Logistic_Regression/server"  # Replace with your application name/path
# $AMD1 -m ipc,fp,l1,l2 -d 300 -o ../Logistic_Regression/Server/perfomance.csv -- "../Logistic_Regression/server" 
# # echo "Started"
# # # # Define the application you want to run
# # # # Loop to run the application 10 times

# kill_port_12344
# AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
# APP="../Linear_Regression/server"  # Replace with your application name/path
# $AMD1 -m ipc,fp,l1,l2 -d 300 -o ../Linear_Regression/Server/perfomance.csv -- "../Linear_Regression/server" 
# # echo "Started"
# # # Define the application you want to run
# # Loop to run the application 10 times

# kill_port_12344
# AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
# APP="../KernelSVM/server"  # Replace with your application name/path
# $AMD1 -m ipc,fp,l1,l2 -d 300 -o ../KernelSVM/Server/perfomance.csv -- "../KernelSVM/server" 
# # echo "Started"
# # Define the application you want to run
# Loop to run the application 10 times

# kill_port_12344
# AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
# APP="../LSVM/server"  # Replace with your application name/path
# $AMD1 -m ipc,fp,l1,l2 -d 300 -o ../LSVM/Server/perfomance.csv -- "../LSVM/server" 
# # echo "Started"
# # # Define the application you want to run
# # # Loop to run the application 10 times

kill_port_12344
AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
APP="../KMeans/server"  # Replace with your application name/path
$AMD1 -m ipc,fp,l1,l2,l3 -d 300 -o ../KMeans/Server/perfomance.csv -- "../KMeans/server" 
# #echo "Started"
# # # Define the application you want to run
# # # Loop to run the application 10 times


# kill_port_12344
# AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
# APP="../Adaboost/server"  # Replace with your application name/path
# $AMD1 -m ipc,fp,l1,l2 -d 300 -o ../Adaboost/Server/perfomance.csv -- "../Adaboost/server" 
# echo "Started"
# Define the application you want to run
# Loop to run the application 10 times

# kill_port_12344
# AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
# APP="../RF/server"  # Replace with your application name/path
# $AMD1 -m ipc,fp,l1,l2 -d 300 -o ../RF/Server/perfomance.csv -- "../RF/server" 
# # echo "Started"
# # Define the application you want to run
# # Loop to run the application 10 times