#!/bin/bash
kill_port_12344() {
    lsof -i:12344 -t | xargs -r kill -9
}

kill_port_12344

# AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
# APP="../Naive_Bayes/client"  # Replace with your application name/path
# $AMD1 -m l3 -d 300 -o ../Naive_Bayes/Client/perfomance.csv -- "../Naive_Bayes/client" &
# # Define the application you want to run
# # Loop to run the application 10 times

# #$APP
# for i in {1..25}
# do  
#     #echo "Running iteration $i..."
#     $APP & # Execute the application
# done
# echo "STARTED"

# AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
# APP="../Logistic_Regression/client"  # Replace with your application name/path
# $AMD1 -m l3 -d 300 -o ../Logistic_Regression/Client/perfomance.csv -- "../Logistic_Regression/client" &
# # Define the application you want to run
# # Loop to run the application 10 times

# #$APP
# for i in {1..25}
# do  
#     #echo "Running iteration $i..."
#     $APP & # Execute the application
# done
# echo "STARTED"

# AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
# APP="../Linear_Regression/client"  # Replace with your application name/path
# $AMD1 -m l3 -d 300 -o ../Linear_Regression/Client/perfomance.csv -- "../Linear_Regression/client" &
# # # Define the application you want to run
# # # Loop to run the application 10 times

# #$APP
# for i in {1..25}
# do  
#     #echo "Running iteration $i..."
#     $APP & # Execute the application
# done
# echo "STARTED"


AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
APP="../KernelSVM/client"  # Replace with your application name/path
$AMD1 -m l3 -d 300 -o ../KernelSVM/Client/perfomance.csv -- "../KernelSVM/client" &
# Define the application you want to run
# Loop to run the application 10 times

#$APP
for i in {1..25}
do  
    #echo "Running iteration $i..."
    $APP & # Execute the application
done
echo "STARTED"

# AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
# APP="../LSVM/client"  # Replace with your application name/path
# $AMD1 -m l3 -d 300 -o ../LSVM/Client/perfomance.csv -- "../LSVM/client" &
# # Define the application you want to run
# # Loop to run the application 10 times

# #$APP
# for i in {1..25}
# do  
#     #echo "Running iteration $i..."
#     $APP & # Execute the application
# done
# echo "STARTED"

# AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
# APP="../KMeans/client"  # Replace with your application name/path
# $AMD1 -m l3 -d 300 -o ../KMeans/Client/perfomance.csv -- "../KMeans/client" &
# # # Define the application you want to run
# # # Loop to run the application 10 times

# #$APP
# for i in {1..29}
# do  
#     #echo "Running iteration $i..."
#     $APP & # Execute the application
# done
# echo "STARTED"

# AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
# APP="../Adaboost/client"  # Replace with your application name/path
# $AMD1 -m l3 -d 300 -o ../Adaboost/Client/perfomance.csv -- "../Adaboost/client" &
# # Define the application you want to run
# # Loop to run the application 10 times

# #$APP
# for i in {1..29}
# do  
#     #echo "Running iteration $i..."
#     $APP & # Execute the application
# done
# echo "STARTED"

# AMD1=/opt/AMDuProf_5.0-1479/bin/AMDuProfPcm
# APP="../RF/client"  # Replace with your application name/path
# $AMD1 -m l3 -d 300 -o ../RF/Client/perfomance.csv -- "../RF/client" &
# # Define the application you want to run
# # Loop to run the application 10 times

# #$APP
# for i in {1..25}
# do  
#     #echo "Running iteration $i..."
#     $APP & # Execute the application
# done
# echo "STARTED"