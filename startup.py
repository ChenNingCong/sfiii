import subprocess
import shlex
import os
def make_command_create_container(id : int):
    return f"apptainer instance start --userns -B /usr/share/alsa -B /usr/bin/getopt -B /usr/bin/cut --bind {os.path.expanduser('~')}/.diambra/credentials:/tmp/.diambra/credentials,{os.getcwd()}/roms:/opt/diambraArena/roms docker://diambra/engine:latest engine_{id}"
def make_command_run_engine(id : int, port : int):
    return f"apptainer exec --userns instance://engine_{id} /bin/diambraEngineServer --envAddress 0.0.0.0:{port}"
def make_command_stop_container(id : int):
    return f"apptainer instance stop engine_{id}"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to create")
parser.add_argument("--start_port", type=int, default=5000, help="Starting port number for the engines")
parser.add_argument("--kill_all", action="store_true", help="Starting port number for the engines")
opt = parser.parse_args()
num_envs = opt.num_envs
start_port = opt.start_port
# Firstly, check whether the ports are available
import socket
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
ports_in_use = []
for i in range(num_envs):
    port = start_port + i
    if is_port_in_use(port):
        ports_in_use.append(port)


for i in range(num_envs):
    stop_cmd = make_command_stop_container(i)
    print(f"Stopping container with command: {stop_cmd}")
    try:
        subprocess.run(shlex.split(stop_cmd), check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while stopping container {i}: {e}")
if not opt.kill_all:
    if ports_in_use:
        print(f"Error: The following ports are already in use: {ports_in_use}")
        print("Please free these ports or choose a different starting port.")
        exit(1)
    import time
    try:
        for i in range(num_envs):
            create_cmd = make_command_create_container(i)
            run_cmd = make_command_run_engine(i, start_port + i)
            print(f"Creating container with command: {create_cmd}")
            subprocess.run(shlex.split(create_cmd), check=True)
            print(f"Running engine with command: {run_cmd}")
            subprocess.Popen(shlex.split(run_cmd))
        print(f"Successfully started {num_envs} engine instances.")
        for i in range(500000000000000):
            time.sleep(1)  # Wait a bit to ensure all engines are up
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
    except KeyboardInterrupt as e:
        print("Interrupted by user, stopping all containers...")
    finally:
        for i in range(num_envs):
            stop_cmd = make_command_stop_container(i)
            print(f"Stopping container with command: {stop_cmd}")
            try:
                subprocess.run(shlex.split(stop_cmd), check=True)
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while stopping container {i}: {e}")