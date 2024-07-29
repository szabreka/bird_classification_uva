#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=14:00:00
#SBATCH -o PredictionLogs/v1_ensemble_prediction_out-%j          # send stdout to outfile
#SBATCH -e PredictionLogs/v1_ensemble_prediction_error-%j          # send stderr to errfile
#SBATCH -J v1_emsemble_prediction

module load 2022
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

python Predict.py "V1"
