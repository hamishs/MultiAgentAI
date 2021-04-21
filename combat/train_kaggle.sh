#!/bin/bash
python3 kaggle_auth.py
python3 train.py --run_name $1 --config_name $2