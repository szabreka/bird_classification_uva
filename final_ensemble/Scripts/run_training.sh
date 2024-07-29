#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=14:00:00
#SBATCH -o TrainingLogs/vit_v1_training_out-%j          # send stdout to outfile
#SBATCH -e TrainingLogs/vit_v1_training_error-%j          # send stderr to errfile
#SBATCH -J v1_train_vit

module load 2022
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

pip freeze > requirements.txt

python Train.py "vit" "v1"