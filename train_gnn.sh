#!/bin/bash
#SBATCH --account=def-boudrea1
#SBATCH --job-name=gnn_train
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --output=logs/gnn_train.out
#SBATCH --error=logs/gnn_train.err
#SBATCH --time=0-4:30:00

module load python scipy-stack
source /project/6099921/NDOT/QML-gates-tests/qqe/.venv/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -u train_gnn.py --epochs=25