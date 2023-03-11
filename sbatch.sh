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
#python run.py --output_name prompt_random --scoring_model gpt2-medium \
#--base_model_name gpt2-xl --mask_filling_model_name t5-large \
#--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
#--dataset squad --dataset_key context --skip_baselines --prompt "Complete this with random words: "
#
#python run.py --output_name prompt_wacky_noun --scoring_model gpt2-medium \
#--base_model_name gpt2-xl --mask_filling_model_name t5-large \
#--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
#--dataset squad --dataset_key context --skip_baselines --prompt "Complete this with wacky nouns: "
#
#python run.py --output_name prompt_angry --scoring_model gpt2-medium \
#--base_model_name gpt2-xl --mask_filling_model_name t5-large \
#--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
#--dataset squad --dataset_key context --skip_baselines --prompt "Complete this with an angry tone: "
#
#
#exit

python run.py --output_name vanilla --scoring_model gpt2-medium \
--base_model_name gpt2-xl --mask_filling_model_name t5-large \
--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset squad --dataset_key context --skip_baselines

python run.py --output_name adj_concentration --scoring_model gpt2-medium \
--base_model_name gpt2-xl --mask_filling_model_name t5-large \
--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset squad --dataset_key context --skip_baselines --concentration "ADJ"

python run.py --output_name noun_concentration --scoring_model gpt2-medium \
--base_model_name gpt2-xl --mask_filling_model_name t5-large \
--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset squad --dataset_key context --skip_baselines --concentration "NOUN"

python run.py --output_name verb_concentration --scoring_model gpt2-medium \
--base_model_name gpt2-xl --mask_filling_model_name t5-large \
--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset squad --dataset_key context --skip_baselines --concentration "VERB"

python run.py --output_name adj_concentration --scoring_model gpt2-medium \
--base_model_name gpt2-xl --mask_filling_model_name t5-large \
--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset squad --dataset_key context --skip_baselines --concentration "ADV"

python run.py --output_name adj_concentration --scoring_model gpt2-medium \
--base_model_name gpt2-xl --mask_filling_model_name t5-large \
--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset squad --dataset_key context --skip_baselines --concentration "PROPN"

exit

#python run.py --output_name partsspeech_AVN_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
#--n_perturbation_list 50,100 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
#--dataset squad --dataset_key context --skip_baselines --concentration "ALL"
#
#
#python run.py --output_name vanilla --base_model_name gpt2-medium --mask_filling_model_name t5-large \
#--n_perturbation_list 50,100 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
#--dataset squad --dataset_key context --skip_baselines


#python run.py --output_name partsspeech_AVN_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
#--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 2 \
#--dataset squad --dataset_key context --skip_baselines --concentration "ALL"
#
#python run.py --output_name partsspeech_AVN_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
#--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
#--dataset squad --dataset_key context --skip_baselines --concentration "ALL"

python run.py --output_name partsspeech_ADJ_conc --base_model_name gpt2-xl --mask_filling_model_name t5-3b \
--n_perturbation_list 5,10 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset squad --dataset_key context --skip_baselines --concentration "ADJ"

python run.py --output_name partsspeech_ADV_conc --base_model_name gpt2-xl --mask_filling_model_name t5-3b \
--n_perturbation_list 5,10 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset squad --dataset_key context --skip_baselines --concentration "ADV" #DOESN'T WORK

python run.py --output_name partsspeech_NOUN_conc --base_model_name gpt2-xl --mask_filling_model_name t5-3b \
--n_perturbation_list 5,10 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset squad --dataset_key context --skip_baselines --concentration "NOUN"

python run.py --output_name partsspeech_FREQ_conc --base_model_name gpt2-xl --mask_filling_model_name t5-3b \
--n_perturbation_list 5,10 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset squad --dataset_key context --skip_baselines --concentration "FREQ"

python run.py --output_name partsspeech_PROPN_conc --base_model_name gpt2-xl --mask_filling_model_name t5-3b \
--n_perturbation_list 5,10 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset squad --dataset_key context --skip_baselines --concentration "PROPN" #  DOESN'T WORK

python run.py --output_name partsspeech_VERB_conc --base_model_name gpt2-xl --mask_filling_model_name t5-3b \
--n_perturbation_list 5,10 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset squad --dataset_key context --skip_baselines --concentration "VERB"

#python run.py --output_name vanilla --base_model_name gpt2-xl --mask_filling_model_name t5-3b \
#--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
#--dataset squad --dataset_key context --skip_baselines
