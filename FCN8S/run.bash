#!/bin/bash
#SBATCH --job-name=FCN
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --output=re_test.out.%j
#SBATCH --error=re_test.err.%j
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
python3 train.py
