import subprocess
import sys

def run_cmd(cmd):
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors/Warnings:\n", result.stderr)

print("======== Checking Streamlit Installation ========")

# 1) Check if streamlit installed
run_cmd([sys.executable, "-m", "pip", "show", "streamlit"])

# 2) Check dependency issues
print("\n======== Checking Dependencies ========")
run_cmd([sys.executable, "-m", "pip", "check"])

# 3) Test running streamlit
print("\n======== Testing Streamlit Run ========")
run_cmd([sys.executable, "-m", "streamlit", "hello"])

print("\n======== DONE ========")