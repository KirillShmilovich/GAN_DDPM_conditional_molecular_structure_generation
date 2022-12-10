#!/bin/bash

#SBATCH --job-name=gen=256_dis=512_latent=128_genNet=simple_disNet=simple_opt=rmsprop_lr=1e-5_LONG
#SBATCH --output=out_files/out_%x.out
#SBATCH --account=pi-andrewferguson
#SBATCH --partition=andrewferguson-gpu
#SBATCH --ntasks-per-node=1 # num cores to drive each gpu 
#SBATCH --cpus-per-task=10   # set this to the desired number of threads
#SBATCH --time=160:00:00
#SBATCH --gres=gpu:1


module load python
srun /project/andrewferguson/Kirill/class_project_env/bin/python PentaPeptideHeavy.py --trainer.logger.class_path=pytorch_lightning.loggers.TensorBoardLogger --trainer.logger.init_args.save_dir="./" --trainer.logger.init_args.name=PentaPeptideHeavySweepLast --trainer.logger.init_args.version=gen=256_dis=512_latent=128_genNet=simple_disNet=simple_opt=rmsprop_lr=1e-5_LONG --model.gen_hidden_dim=256 --model.dis_hidden_dim=512 --model.latent_dim=128 --model.gen_network_type=simple --model.dis_network_type=simple --model.opt=rmsprop --model.lr=1e-5
