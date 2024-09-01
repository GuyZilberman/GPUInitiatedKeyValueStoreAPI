import subprocess

def run_command(command):
    """Run a shell command and stream the output, returning the full log."""
    print(f"Running: {command}")
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
        print(f"Wrong answer detected during {context}, aborting.")
        return True
    return False

def run_all_tests(mode):
    print("=============================================================")
    # Run the program with async option
    success, log = run_command("./kvapp --tb 1 --rk async")
    if not success or check_for_wrong(log, mode + " "  + "async"):
        exit(1)
    
    print("=============================================================")
    # Run the program with sync option
    success, log = run_command("./kvapp --tb 1 --rk sync")
    if not success or check_for_wrong(log, mode + " "  + "sync"):
        exit(1)
    
    print("=============================================================")
    # Run the program with async option and host writes
    success, log = run_command("./kvapp --w host --tb 1 --rk async")
    if not success or check_for_wrong(log, mode + " "  + "host writes + async"):
        exit(1)
    
    print("=============================================================")
    # Run the program with sync option and host writes
    success, log = run_command("./kvapp --w host --tb 1 --rk sync")
    if not success or check_for_wrong(log, mode + " "  + "host writes + sync"):
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
    print("All commands executed successfully without any wrong answers in the logs.")

if __name__ == "__main__":
    main()
