# kill all processes related to main.py for the current user
pkill -f -u "$(whoami)" 'python.*main.py'
# kill all spwaned forkserver processes for the current user
pkill -f -u "$(whoami)" 'python3.*multiprocessing.forkserver'
pkill -f -u "$(whoami)" 'python.*main.py'
lsof -nP -iTCP -sTCP:LISTEN |grep diam|wc -l
rm *.lock
conda activate minerl
CUDA_VISIBLE_DEVICES=0 BASE_PORT=50000 python3 main.py --cfgFile config/config_long_high_gamma.yaml &
conda activate minerl
CUDA_VISIBLE_DEVICES=1 BASE_PORT=50016 python3 main.py --cfgFile config/config_long.yaml &
conda activate minerl
CUDA_VISIBLE_DEVICES=2 BASE_PORT=50032 python3 main.py --cfgFile config/config_long.yaml &