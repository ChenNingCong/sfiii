import subprocess
import shlex
import os
import argparse
import socket
import time

def make_command_run_container(id: int, port: int):
    # Since the binary is the ENTRYPOINT in your patched image, 
    # we pass the arguments directly to 'docker run'
    return (
        f"docker run -d --name engine_{id} "
        f"-p {port}:{port} "
        f"-v {os.getcwd()}/roms:/opt/diambraArena/roms "
        f"diambra/engine:patched "
        f"--envAddress 0.0.0.0:{port}"
    )

def make_command_stop_container(id: int):
    # Removed double return; use -f to force remove running containers
    return f"docker rm -f engine_{id}"

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8, help="Number of environments")
    parser.add_argument("--start_port", type=int, default=5000, help="Starting port")
    parser.add_argument("--kill_all", action="store_true", help="Only kill existing containers")
    opt = parser.parse_args()

    # 1. Cleanup & Port Check
    ports_in_use = []
    for i in range(opt.num_envs):
        # Always try to remove existing container first to avoid name conflicts
        stop_cmd = make_command_stop_container(i)
        subprocess.run(shlex.split(stop_cmd), capture_output=True)
        
        if not opt.kill_all and is_port_in_use(opt.start_port + i):
            ports_in_use.append(opt.start_port + i)

    if opt.kill_all:
        print("All containers stopped.")
        return

    if ports_in_use:
        print(f"Error: Ports already in use: {ports_in_use}")
        exit(1)

    # 2. Launching
    processes = []
    try:
        for i in range(opt.num_envs):
            run_cmd = make_command_run_container(i, opt.start_port + i)
            print(f"Launching environment {i} on port {opt.start_port + i}...")
            
            # Using shell=False with shlex.split is safer for simple commands
            subprocess.run(shlex.split(run_cmd), check=True)
        
        print(f"Successfully started {opt.num_envs} instances. Press Ctrl+C to stop.")
        
        # Keep the script alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user, cleaning up...")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start container: {e}")
    finally:
        # 3. Final Cleanup
        for i in range(opt.num_envs):
            subprocess.run(shlex.split(make_command_stop_container(i)), capture_output=True)

if __name__ == "__main__":
    main()