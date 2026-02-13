#!/bin/bash
#SBATCH --account=def-boudrea1
#SBATCH --job-name=n_layers_sweep
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --output=logs/n_layers_sweep.out
#SBATCH --error=logs/n_layers_sweep.err
#SBATCH --time=0-0:30:00

module load python scipy-stack
source /project/6099921/NDOT/QML-gates-tests/qqe/.venv/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python -u main.py sweep 10 25 --repeat=15 --sweep-type="tcount"