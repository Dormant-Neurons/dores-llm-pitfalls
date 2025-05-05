#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./outputs/%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J PT_GPU
# Queue:
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40000
#
#SBATCH --mail-type=all
#SBATCH --mail-user=niklas.risse@mpi-sp.org
#
# wall clock limit
#SBATCH --time=04:00:00

module purge
module load anaconda/3/2021.11
module load cuda/11.6
module load pytorch/gpu-cuda-11.6/1.12.0
module load scikit-learn/1.1.1

python -m pip install --user --quiet \
       transformers datasets evaluate accelerate \
       tqdm wandb h5py dill

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

srun python dores-llm-pitfalls/pitfall_5/eval_experiment.py $*
echo "job finished"