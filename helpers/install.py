import subprocess
import sys


def install_requirements():
    """Install the requirements.txt file."""
    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "-r", "requirements.txt"])
