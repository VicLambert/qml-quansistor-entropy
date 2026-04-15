#!/bin/bash
#SBATCH --account=def-boudrea1
#SBATCH --job-name=predictions
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --output=logs/prediction/global.out
#SBATCH --error=logs/prediction/global.err
#SBATCH --time=0-4:30:00

module load python scipy-stack
source /project/6099921/NDOT/QML-gates-tests/qqe/.venv/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -u predictions.py --model-kind nn --training-scope global
