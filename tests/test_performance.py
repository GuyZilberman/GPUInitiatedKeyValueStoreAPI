import subprocess
import pandas as pd
import numpy as np
import time
import os
import argparse
from datetime import datetime
import csv
import yaml

SLEEP_TIME = 10

# Set up argument parser
parser = argparse.ArgumentParser(description='Run performance tests with different configurations.')
parser.add_argument('--mode', type=str, choices=['XDP', 'IN_MEMORY_STORE', 'STORELIB_LOOPBACK'], required=True,
                    help='Mode to run the tests in. Choose from XDP, IN_MEMORY_STORE, or STORELIB_LOOPBACK.')
parser.add_argument('--rk', type=str, choices=['sync', 'async'], required=True,
                    help='Read kernel mode to run the tests in. Choose from sync or async.')

args = parser.parse_args()

# Set the appropriate flags based on the mode
USE_IN_MEMORY_STORE = args.mode == 'IN_MEMORY_STORE'
USE_STORELIB_LOOPBACK = args.mode == 'STORELIB_LOOPBACK'

# Extract the rk argument
rk_mode = args.rk

# Define the range of VALUE_SIZEs and thread blocks
value_sizes = [512]
thread_blocks = [1]
NUM_KEYS = 512
NUM_RUNS_PER_TB_SIZE = 3


# This function will run a command and return its output
def run_command(command, print_output=False):
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
        if print_output:
            print(output)
        return output
    except subprocess.CalledProcessError as e:
        print(f"Command failed with {e.returncode}: {e.output}")
        exit(1)

# Load data from the YAML file
def load_yaml_data(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data

# Save metadata excluding 'numThreadBlocks'
def save_metadata(yaml_data, metadata_file):
    settings = yaml_data.get('Settings', {})
    if 'numThreadBlocks' in settings:
        settings.pop('numThreadBlocks')
    with open(metadata_file, 'w') as file:
        yaml.dump({'Settings': settings}, file)

# Determine the directory name based on script settings and current date-time
def directory_name():
    XDP_header_path = "/etc/pliops/store_lib_expo.h"
    XDP_on_host_header_path = "/etc/opt/pliops/xdp-onhost/store_lib_expo.h"
    current_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    dir_name = f"results_keys_{NUM_KEYS}_{current_time}_{rk_mode.upper()}"
    if USE_IN_MEMORY_STORE:
        dir_name += "_IN_MEMORY_STORE"
    elif USE_STORELIB_LOOPBACK:
        dir_name += "_STORELIB_LOOPBACK"
    elif os.path.exists(XDP_header_path):
        dir_name += "_XDP"
    elif os.path.exists(XDP_on_host_header_path):
        dir_name += "_XDP_ON_HOST"
    else:
        print("No mode was found in directory_name()")
    return dir_name

# Create directory if it does not exist
output_dir = directory_name()
os.makedirs(output_dir, exist_ok=True)

# Metadata file to save settings
metadata_file = os.path.join(output_dir, 'metadata.yaml')

# Iterate over each VALUE_SIZE
for size in value_sizes:
    print("========================================")
    print(f"Starting run for VALUE_SIZE={size}")
    start_time = time.perf_counter()

    # Define the CSV file path for the current VALUE_SIZE
    file_path = os.path.join(output_dir, f"results_{size}.csv")

    # Write headers to the CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Thread Blocks", "Write Time (s)", "Read Time (s)", "Write Bandwidth (GB/s)", "Read Bandwidth (GB/s)", "Write IOPS", "Read IOPS"])

    results = []
    
    # Build the application with the current VALUE_SIZE
    build_commands = [
        "make clean",
        f"bear make VALUE_SIZE={size} NUM_KEYS={NUM_KEYS} -j"
    ]

    if USE_IN_MEMORY_STORE and USE_STORELIB_LOOPBACK:
         print("Both USE_IN_MEMORY_STORE and USE_STORELIB_LOOPBACK are set to True. Exiting.")
         exit(1)     
    if USE_IN_MEMORY_STORE:
            build_commands[1] += " IN_MEMORY_STORE=1"
    elif USE_STORELIB_LOOPBACK:
            build_commands[1] += " STORELIB_LOOPBACK=1"

    for cmd in build_commands:
        run_command(cmd, print_output=True)
    
    first_run = True
    
    # Run the application with each --tb value
    for tb in thread_blocks:
        write_times = []
        read_times = []
        write_bandwidths = []
        read_bandwidths = []
        write_iops = []
        read_iops = []
        
        # Run kvapp several times
        for _ in range(NUM_RUNS_PER_TB_SIZE):
            command = f"sudo -E ./kvapp --tb {tb} --rk {rk_mode}"
            run_command(command)

            # Load the results from YAML file
            yaml_data = load_yaml_data('log_output.yaml')

            if first_run:
                save_metadata(yaml_data, metadata_file)
                first_run = False

            write_results = yaml_data.get('write_kernel', {})
            read_results = yaml_data.get('read_kernel', {}) if rk_mode == 'sync' else yaml_data.get('async_read_kernel_3phase', {})

            # Extract and store necessary output from the YAML data
            write_times.append(float(write_results.get('elapsed_time [s]', 0)))
            read_times.append(float(read_results.get('elapsed_time [s]', 0)))
            write_bandwidths.append(float(write_results.get('effective_bandwidth [GB/s]', 0)))
            read_bandwidths.append(float(read_results.get('effective_bandwidth [GB/s]', 0)))
            write_iops.append(float(write_results.get('IOPS', 0)))
            read_iops.append(float(read_results.get('IOPS', 0)))

            # Sleep to allow the XDP to be ready
            time.sleep(SLEEP_TIME)
        
        # Calculate the median of times, bandwidths, and IOPS
        median_write_time = np.median(write_times)
        median_read_time = np.median(read_times)
        median_write_bandwidth = np.median(write_bandwidths)
        median_read_bandwidth = np.median(read_bandwidths)
        median_write_iops = np.median(write_iops)
        median_read_iops = np.median(read_iops)
        
        # Prepare the row of results
        result_row = [tb, median_write_time, median_read_time, median_write_bandwidth, median_read_bandwidth, median_write_iops, median_read_iops]
        
        # Append the row to the CSV file
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(result_row)

        print(f"Finished running tb = {tb}, size = {size}")
    
    end_time = time.perf_counter()

    # Calculate the duration
    duration_in_minutes = (end_time - start_time) / 60 
    print("Time taken to finish:", duration_in_minutes, "minutes")
    print(f"Results for VALUE_SIZE={size} saved successfully in {file_path}.")
