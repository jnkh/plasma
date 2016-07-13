#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --ntasks-per-socket=8
#SBATCH --gres=gpu:2
#SBATCH -t 3-0:00

cd ~/plasma/src
python learn.py
