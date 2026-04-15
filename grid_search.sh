#!/bin/bash
#SBATCH --account=def-boudrea1
#SBATCH --job-name=grid_search
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --output=logs/grid_search.out
#SBATCH --error=logs/grid_search.err
#SBATCH --time=2-18:00:00

module load python scipy-stack
source /project/6099921/NDOT/QML-gates-tests/qqe/.venv/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -u grid_search_gnn.py --epochs=50 
