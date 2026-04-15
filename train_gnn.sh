#!/bin/bash
#SBATCH --account=def-boudrea1
#SBATCH --job-name=nn_train
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --output=logs/nn_train.out
#SBATCH --error=logs/nn_train.err
#SBATCH --time=0-2:30:00

module load python scipy-stack
source /project/6099921/NDOT/QML-gates-tests/qqe/.venv/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -u train_gnn.py --epochs=50 --loss-type huber --model-type nn
