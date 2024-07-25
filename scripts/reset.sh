#!/bin/bash

# Function to print and run a command
run_command() {
    echo "$@"
    "$@"
}

# Stopping the service
run_command sudo systemctl stop pliostore@0.service

# Running the pliocli support command
run_command sudo pliocli support start_next_clean -f

# Starting the service
run_command sudo systemctl start pliostore@0.service

echo "All commands executed successfully."
