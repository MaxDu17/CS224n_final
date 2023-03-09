#!/bin/bash
#!/bin/bash
#python run.py --output_name partsspeech_ADV_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
#--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.1 --span_length 1 \
#--dataset squad --dataset_key context --skip_baselines --concentration "ADV" #DOESN'T WORK

#python run.py --output_name partsspeech_FREQ_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
#--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 2 \
#--dataset squad --dataset_key context --skip_baselines --concentration "FREQ"


python run.py --output_name vanilla \
--base_model_name gpt2-medium --mask_filling_model_name t5-large \
--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset squad --dataset_key context --skip_baselines --prompt "Complete this with wrong spelling: "

exit

python run.py --output_name vanilla --scoring_model gpt2-medium \
--base_model_name gpt2-xl --mask_filling_model_name t5-large \
--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset squad --dataset_key context --skip_baselines


exit

python run.py --output_name partsspeech_PROPN_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 2 \
--dataset squad --dataset_key context --skip_baselines --concentration "PROPN"


python run.py --output_name partsspeech_ADJ_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 2 \
--dataset squad --dataset_key context --skip_baselines --concentration "ADJ"

python run.py --output_name partsspeech_stanza_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 2 \
--dataset squad --dataset_key context --skip_baselines --concentration "ALL"

python run.py --output_name partsspeech_FREQ_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 2 \
--dataset squad --dataset_key context --skip_baselines --concentration "FREQ"

#python run.py --output_name vanilla --base_model_name gpt2-medium --mask_filling_model_name t5-large \
#--n_perturbation_list 5 --n_samples 312 --pct_words_masked 0.1 --span_length 2 \
#--dataset squad --dataset_key context --skip_baselines
