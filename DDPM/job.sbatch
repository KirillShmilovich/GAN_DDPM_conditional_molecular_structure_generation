#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikejones@uchicago.ed

#SBATCH --partition=gpu #andrewferguson-gpu
#SBATCH --account=pi-andrewferguson
#SBATCH --nodes=1 # SET NUM NODES
#SBATCH --ntasks-per-node=1  # SETS NUM MPI RANKS (1 PER GPU)
##SBATCH --cpus-per-task=8  # SET NUM THREADS (cores)
#SBATCH --mem=20GB # REQUEST MEMORY
#SBATCH --gres=gpu:1 # SET NUM GPUS *
#SBATCH --exclude=midway3-0294

#SBATCH --job-name=diffusion
#SBATCH --output=job.out

NTASKS=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load openmpi
module load python
module load cuda

python run_training_MJ.py --model Unet1D --save adp_Unet1D_500000s --beta cosine --loss l1 --nsteps 500000 --train train_adp --cond test_adp_cond --natoms 8 --nsrvs 2