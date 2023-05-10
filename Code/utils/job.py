#!/usr/bin/python
#
import os
import subprocess
import sys
import shutil

try:
    from config import load_config
except:
    from utils.config import load_config

_JOB_NAME = "cei_complex_gan"
_MODEL_NAME = "GAN"


def get_command_line(toplogdir, config_file_path=""):
    """
    Return the command line to execute in the job.
    """
    command = sys.argv

    # Change top log directory
    if "--toplogdir" in command:
        idx = command.index("--toplogdir")
        command[idx + 1] = toplogdir
    elif "-lgd" in command:
        idx = command.index("-lgd")
        command[idx + 1] = toplogdir
    else:
        command.append("--toplogdir")
        command.append(toplogdir)

    # Change config file path
    if config_file_path != "":
        if "-cfg" in command:
            idx = command.index("-cfg")
            command[idx + 1] = config_file_path
        elif "--config" in command:
            idx = command.index("--config")
            command[idx + 1] = config_file_path
        else:
            command.append("--config")
            command.append(config_file_path)

    command.remove("--job")
    command = ["python3"] + command
    command_line = " ".join(command)
    return command_line


def get_commit_id():
    # Ensure all the modified files have been staged and commited
    result = int(
        subprocess.run(
            "expr $(git diff --name-only | wc -l) + $(git diff --name-only --cached | wc -l)",
            shell=True,
            stdout=subprocess.PIPE,
        ).stdout.decode()
    )
    if result > 0:
        print(f"We found {result} modifications either not staged or not commited")
        raise RuntimeError(
            "You must stage and commit every modification before submission "
        )

    commit_id = subprocess.check_output(
        "git log --pretty=format:'%H' -n 1", shell=True
    ).decode()
    return commit_id


def get_git_command_line(commit_last=None):
    # Code has to be up to date
    if commit_last is not None:
        commit_id = get_commit_id()
        git_command_line = f"""
                            echo "Checking out the correct version of the code commit_id {commit_id}"
                            cd $TMPDIR/zooplankton/
                            git checkout {commit_id}
                            """
    else:
        git_command_line = ""
    return git_command_line


def mkdir_if_not_exists(path):
    if not (os.path.exists(path)):
        os.mkdir(path)


def makejob(args, commit_last=None):
    """
    Launch sbatch jobs.
    Args:
        - model_name: name of the model
        - config_path: path of the config file
        - nruns: number of runs to launch
        - commit_last: Code has to be commited and up to date if not None.
    """
    cfg = load_config(args.config)

    # Get absolute path of log dir
    pwd = os.getcwd()
    toplogdir = cfg["TRAIN"]["LOGGING"]["TOP_LOG_DIR"].replace("./", "")
    logs_path = os.path.join(pwd, toplogdir)

    if not (os.path.exists(logs_path)):
        assert os.path.exists(
            args.toplogdir
        ), f"Can't find logdir {args.toplogdir}. Please enter a valid logs directory."

    model_name = _MODEL_NAME
    nruns = args.nruns

    # Logs slurm
    i = 0
    while True:
        run_name = cfg["TRAIN"]["LOGGING"]["RUN_NAME"] + "_" + str(i)
        run_path = os.path.join(logs_path, run_name)
        if not (os.path.isdir(run_path)):
            break
        i += 1

    logs_slurm_path = os.path.join(toplogdir, run_name)
    mkdir_if_not_exists(logs_slurm_path)

    # Copy config file
    config_file_path = os.path.join(logs_path, run_name, "config.yaml")
    shutil.copy(args.config, config_file_path)

    command_line = get_command_line(logs_path, config_file_path)
    git_command_line = get_git_command_line(commit_last)

    command_line = f"{command_line} --run_name {run_name}"

    print(f"Launching job with:\n{command_line}")

    return f"""#!/bin/bash

#SBATCH --job-name={_JOB_NAME}-{model_name}
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output={logs_slurm_path}/slurm-%A_%a.out
#SBATCH --error={logs_slurm_path}/slurm-%A_%a.err
#SBATCH --array=0-{nruns}
#SBATCH --exclude=sh[10-16]

current_dir=`pwd`

echo "Session " {model_name}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
my_directory={model_name}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
echo "Copying the source directory and data"
pwd
date
mkdir $TMPDIR/{_JOB_NAME}
rsync -r --exclude 'logs/*' --exclude 'env/*' --exclude 'venv/*' --exclude 'wandb/*' . $TMPDIR/{_JOB_NAME}/

{git_command_line}

echo "Setting up the virtual environment"
python3 -m pip install virtualenv
virtualenv -p python3 $TMPDIR/venv
source $TMPDIR/venv/bin/activate
python -m pip install -r requirements.txt

echo "Training"
nvidia-smi -q | grep "CUDA Version"

{command_line}

if [[ $? != 0 ]]; then
    exit -1
fi
"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")
    os.remove("job.sbatch")
