#!/bin/bash
#SBATCH --job-name=oregon-seaside.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --output=oregon-seaside.out
#SBATCH --time=01:00:00
#SBATCH --mail-user=aachowdhury2@sheffield.ac.uk
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

module load Python/3.10.4-GCCcore-11.3.0 CUDA/10.1.243

cd /users/cip19aac/LISFLOOD-FP/build/oregon-seaside

./../lisflood -epsilon $1 -dirroot $2 oregon-seaside.par
