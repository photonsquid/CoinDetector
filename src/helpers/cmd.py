# execute shell commands

import subprocess


def run(command, show_output=False):
    """Execute a shell command and return its output"""
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if error:
        error = error.decode('utf-8')
        print(error)
        raise Exception(error)
    output = output.decode('utf-8')
    if show_output:
        print(output)
    return output


def git_update(branch: str = "main", force: bool = False, show_output: bool = False):
    """Update the git repository"""
    run('git fetch --all', show_output)
    if force:
        run('git checkout {}'.format(branch), show_output)
        run('git reset --hard', show_output)
    run('git pull', show_output)


def print_gpu_name():
    """Print the name of the GPU"""
    return run('nvidia-smi --query-gpu=name --format=csv,noheader')
