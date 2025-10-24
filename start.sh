mkdir ~/.diambra
touch ~/.diambra/credentials
module load apptainer
# Output for example: DIAMBRA_ENVS=localhost:8000 localhost:8001 localhost:8002
python startup.py --num_envs $NUM_ENV --start_port $BASE_PORT