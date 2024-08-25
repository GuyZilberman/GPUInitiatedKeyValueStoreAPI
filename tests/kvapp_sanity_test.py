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

def print_separator():
    """Print a separator for better readability."""
    print("=" * 60)

def main():
    # Compile with first set of flags
    print_separator()
    success, log = run_command("make -j CHECK_WRONG_ANSWERS=1")
    if not success:
        print("Error occurred during first make command")
        return
    
    # Run the program with async option
    print_separator()
    success, log = run_command("./kvapp --tb 1 --rk async")
    if not success or check_for_wrong(log, "XDP async"):
        return
    
    # Run the program with sync option
    print_separator()
    success, log = run_command("./kvapp --tb 1 --rk sync")
    if not success or check_for_wrong(log, "XDP sync"):
        return
    
    # Compile with second set of flags
    print_separator()
    success, log = run_command("make -j CHECK_WRONG_ANSWERS=1 IN_MEMORY_STORE=1")
    if not success:
        print("Error occurred during second make command")
        return
    
    # Run the program with async option again
    print_separator()
    success, log = run_command("./kvapp --tb 1 --rk async")
    if not success or check_for_wrong(log, "in-memory async"):
        return
    
    # Run the program with sync option again
    print_separator()
    success, log = run_command("./kvapp --tb 1 --rk sync")
    if not success or check_for_wrong(log, "in-memory sync"):
        return

    print_separator()
    print("All commands executed successfully without any wrong answers in the logs.")

if __name__ == "__main__":
    main()
