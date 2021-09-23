#!/bin/bash -l

#SBATCH --partition=gpu-k40c
#SBATCH --job-name=cifargan7_400epochs
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --ntasks=4
#SBATCH --qos=normal
#SBATCH --nodelist=gpu04

#SBATCH --mail-type=ALL
#SBATCH --mail-user=s2003824@siswa.um.edu.my
#SBATCH --gres=gpu:k40c:2

#gres=gpu:k40c:2
#gres=gpu:gtx1080ti:1
#gres=gpu:titanxp:2
#gres=gpu:k10:8

# Loading Required module
module purge
module load miniconda/miniconda3
#module load cudnn/cudnn-8.1.1/cuda-11.2

#cd /lustre/user/jamesang/
source activate ~/tf2_env/

python tf_gan_cifar_tf2.py -epoch=150 -filenum=1
#python generate_images.py -n_samples=10 -latent_dim=100
