import subprocess

def run_command(command):
    """Run a shell command and stream the output, returning the full log."""
    print(f"\033[95m{command}\033[0m")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    full_log = ""
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            full_log += output
    
    stderr = process.communicate()[1]
    
    if process.returncode != 0:
        print(f"Error: {stderr.strip()}")
        return False, full_log
    
    return True, full_log

def check_for_wrong(log, context):
    """Check if the word 'wrong' is in the log."""
    if "wrong" in log.lower():
        print(f"\033[91mWrong answer detected during {context}, aborting.\033[0m")
        return True
    return False

def run_all_tests(mode):
    read_kernels = ["sync", "async"]
    write_kernels = ["sync", "async"]
    write_modes = ["device", "host"]

    for read_kernel in read_kernels:
        for write_kernel in write_kernels:
            for write_mode in write_modes:
                print("=============================================================")
                # Construct the command based on the kernels
                if write_mode == "host":
                    command = f"./kvapp --tb 1 --w {write_mode} --rk {read_kernel}"
                else: # device
                    command = f"./kvapp --tb 1 --w {write_mode} --wk {write_kernel} --rk {read_kernel}"
                
                # Run the command and check the result
                success, log = run_command(command)
                if not success or check_for_wrong(log, f"{mode} {write_mode} writes + {read_kernel}"):
                    exit(1)

def main():
    print("=============================================================")
    # Compile with first set of flags
    success, _ = run_command("make -j CHECK_WRONG_ANSWERS=1")
    if not success:
        print("Error occurred during first make command (XDP mode)")
        exit(1)
    
    run_all_tests("XDP")
        
    print("=============================================================")
    # Compile with second set of flags
    success, _ = run_command("make -j CHECK_WRONG_ANSWERS=1 IN_MEMORY_STORE=1")
    if not success:
        print("Error occurred during second make command (IN_MEMORY_STORE mode)")
        exit(1)
    
    run_all_tests("IN_MEMORY_STORE")

    print("=============================================================")
    print("\033[92mAll commands executed successfully without any wrong answers in the logs.\033[0m")

if __name__ == "__main__":
    main()
