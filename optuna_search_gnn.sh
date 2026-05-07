#!/bin/bash
#SBATCH --account=def-boudrea1
#SBATCH --job-name=gnn_global_optuna
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --output=logs/gnn_global_optuna.out
#SBATCH --error=logs/gnn_global_optuna.err
#SBATCH --time=1-18:00:00

module load python scipy-stack
source /project/6099921/NDOT/QML-gates-tests/qqe/.venv/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -u optuna_search.py --model-type gnn --training-mode global --epochs 25 --n-trials 20 --study-name gnn_global_optuna --storage sqlite:///optuna_gnn_global.db
