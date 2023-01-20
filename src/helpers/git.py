# execute shell commands

import subprocess


def execute(command, show_output=False):
    """Execute a shell command and return its output"""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if show_output:
        print(output)
    return output


def git_update(branch: str = "main", force: bool = False, show_output: bool = False):
    """Update the git repository"""
    execute('git fetch --all', show_output)
    execute('git checkout {}'.format(branch), show_output)
    if force:
        execute('git reset --hard', show_output)
    execute('git pull', show_output)
