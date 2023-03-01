#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --account=iris

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1 --constraint=24G
#SBATCH --exclude=iris2

#SBATCH --job-name=""
#SBATCH --output=clusterlogs/clusteroutput%j.out
#SBATCH --cpus-per-task=8

# only use the following if you want email notification
#SBATCH --mail-user=maxjdu@stanford.edu
#SBATCH --mail-type=ALL

#SBATCH --time=3-0:0

source ~/.bashrc
conda activate cs330
cd /iris/u/maxjdu/Repos/CS224n_final

echo $SLURM_JOB_GPUS

echo "test"
#python run.py --output_name normal --base_model_name gpt2-medium --mask_filling_model_name t5-large \
#--n_perturbation_list 5 --n_samples 312 --pct_words_masked 0.3 --span_length 2 \
#--dataset squad --dataset_key context --skip_baselines
