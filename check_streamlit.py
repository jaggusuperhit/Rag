import pkg_resources
import sys

# Check if streamlit is installed
try:
    streamlit_version = pkg_resources.get_distribution("streamlit").version
    print(f"Streamlit is installed. Version: {streamlit_version}")
except pkg_resources.DistributionNotFound:
    print("Streamlit is not installed.")
    sys.exit(1)

# Print Python version
print(f"Python version: {sys.version}")

# Print the path to the Python executable
print(f"Python executable: {sys.executable}")

# List all installed packages
print("\nInstalled packages:")
for package in pkg_resources.working_set:
    print(f"{package.key} {package.version}")
