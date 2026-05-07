#!/bin/bash
#SBATCH --account=def-boudrea1
#SBATCH --job-name=nn_global_optuna
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --output=logs/nn_global_optuna.out
#SBATCH --error=logs/nn_global_optuna.err
#SBATCH --time=1-18:00:00

module load python scipy-stack
source /project/6099921/NDOT/QML-gates-tests/qqe/.venv/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -u optuna_search.py --model-type nn --training-mode global --epochs 25 --n-trials 20 --study-name nn_global_optuna --storage sqlite:///optuna_nn_global.db
