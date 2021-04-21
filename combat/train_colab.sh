#!/bin/sh
pip install wandb
wandb login
python3 train.py --run_name $1 --config_name $2