#!/bin/bash -l

#SBATCH --partition=cpu-epyc
#SBATCH --job-name=cifargan7_test7_multithreading
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --nodes=2
#SBATCH --mem=248G
#SBATCH --ntasks=192
#SBATCH --qos=normal
#SBATCH --nodelist=cpu14,cpu15

#SBATCH --mail-type=ALL
#SBATCH --mail-user=s2003824@siswa.um.edu.my


#gres=gpu:k40c:2
#gres=gpu:gtx1080ti:1
#gres=gpu:titanxp:2
#gres=gpu:k10:8

# Loading Required module
module purge
module load miniconda/miniconda3
#module load cudnn/cudnn-8.1.1/cuda-11.2

#cd /scratch/jamesang/
source activate ~/tf2_env/
#cd proj_files/cifar_dcgan4/
python generate_images.py -n_samples=50000 -latent_dim=100 -filename=complete_saved_model_remote_2021-09-22
