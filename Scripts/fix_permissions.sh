#!/bin/bash

# Directory paths
PARENT_DIR="/mnt/combined/home"

# Get all users in the combined_access group
USERS=$(getent group combined_access | awk -F: '{print $4}' | tr ',' ' ')

for user in $USERS; do
    echo "Checking $user..."
    
    # Fix ownership
    sudo chown -R $user:$user $PARENT_DIR/$user
    
    # Fix permissions
    sudo chmod 750 $PARENT_DIR/$user
    
    echo "Fixed permissions for $user"
done

# Ensure parent directory permissions are correct
sudo chgrp combined_access /mnt/combined
sudo chmod 770 /mnt/combined
sudo chmod 755 /mnt/combined/home

echo "Parent directory permissions fixed."
