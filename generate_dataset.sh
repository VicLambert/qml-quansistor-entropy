#!/bin/bash
#SBATCH --account=def-boudrea1
#SBATCH --job-name=generate_dataset_15
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --output=logs/generate_dataset_15.out
#SBATCH --error=logs/generate_dataset_15.err
#SBATCH --time=0-3:30:00

module load python scipy-stack
source /project/6099921/NDOT/QML-gates-tests/qqe/.venv/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -u generate_dataset.py