#!/bin/bash
#SBATCH --account=def-boudrea1
#SBATCH --job-name=generate_data_haar
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --output=qqe/logs/generate_data_haar.out
#SBATCH --error=qqe/logs/generate_data_haar.err
#SBATCH --time=0-3:30:00

module load python scipy-stack
source /project/6099921/NDOT/QML-gates-test/.venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python generate_data.py