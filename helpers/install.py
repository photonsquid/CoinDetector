import subprocess
import sys


def install_requirements():
    """
    Install the requirements.txt file.

    This is a helper function to install the requirements.txt file
    in the current environment.
    """
    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "-r", "requirements.txt"])
