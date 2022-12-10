#!/bin/bash

#SBATCH --job-name=VERSION
#SBATCH --output=out_files/out_%x.out
#SBATCH --account=pi-andrewferguson
#SBATCH --partition=andrewferguson-gpu
#SBATCH --ntasks-per-node=1 # num cores to drive each gpu 
#SBATCH --cpus-per-task=10   # set this to the desired number of threads
#SBATCH --time=160:00:00
#SBATCH --gres=gpu:1


module load python
srun /project/andrewferguson/Kirill/class_project_env/bin/python PentaPeptideHeavy.py --trainer.logger.class_path=pytorch_lightning.loggers.TensorBoardLogger --trainer.logger.init_args.save_dir="./" --trainer.logger.init_args.name=NAME --trainer.logger.init_args.version=VERSION ARGS
