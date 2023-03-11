#!/bin/bash
#!/bin/bash
#python run.py --output_name partsspeech_ADV_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
#--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.1 --span_length 1 \
#--dataset squad --dataset_key context --skip_baselines --concentration "ADV" #DOESN'T WORK

#python run.py --output_name partsspeech_FREQ_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
#--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 2 \
#--dataset squad --dataset_key context --skip_baselines --concentration "FREQ"

## TESTING CHATGPT
# JUST TESTING; WILL USE A SMALLER MODEL LATER
python run.py --output_name chatgpt --batch_size 5 \
--chatgpt --mask_filling_model_name t5-3b \
--scoring_model EleutherAI/gpt-j-6B \
--n_perturbation_list 5,10,20 --n_samples 100 \
--pct_words_masked 0.3 --span_length 2 \
--dataset writing --skip_baselines
exit


### TESTING GPT-J
python run.py --output_name vanilla --scoring_model gpt2-medium \
--base_model_name EleutherAI/gpt-j-6B --mask_filling_model_name t5-3b \
--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset writing --skip_baselines \
#--dataset squad --dataset_key context --skip_baselines


exit

## TESTING THE GPT3 STUFF
python run.py --output_name openai --batch_size 5 \
--openai_model davinci --mask_filling_model_name t5-large \
--scoring_model gpt2-medium \
--n_perturbation_list 5,10 --n_samples 150 \
--pct_words_masked 0.3 --span_length 2 \
--do_top_p --top_p 0.9 --mask_top_p 0.95 \
--dataset writing
#--dataset squad --dataset_key context

exit


### TESTING THE PROMPT STUFF
python run.py --output_name prompt_novel --scoring_model gpt2-medium \
--base_model_name gpt2-xl --mask_filling_model_name t5-large \
--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset squad --dataset_key context --skip_baselines --prompt "I licked the cat. I drank from the moon. The ship sailed on the breadcrumbs. Surf me through the crackle of the night. "

exit

### TESTING CROSS MODEL
python run.py --output_name vanilla --scoring_model gpt2-medium \
--base_model_name gpt2-xl --mask_filling_model_name t5-large \
--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset squad --dataset_key context --skip_baselines


exit

##  TESTING THE CONCENTRATION STUFF
=======
python run.py --output_name STOP_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 2 \
--dataset squad --dataset_key context --skip_baselines --concentration "STOP"

python run.py --output_name NONSTOP_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 2 \
--dataset squad --dataset_key context --skip_baselines --concentration "NONSTOP"

python run.py --output_name FREQ_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 2 \
--dataset squad --dataset_key context --skip_baselines --concentration "FREQ"

python run.py --output_name partsspeech_PROPN_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 2 \
--dataset squad --dataset_key context --skip_baselines --concentration "PROPN"

python run.py --output_name partsspeech_ADJ_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 2 \
--dataset squad --dataset_key context --skip_baselines --concentration "ADJ"

python run.py --output_name partsspeech_stanza_conc --base_model_name gpt2-medium --mask_filling_model_name t5-large \
--n_perturbation_list 5 --n_samples 150 --pct_words_masked 0.3 --span_length 2 \
--dataset squad --dataset_key context --skip_baselines --concentration "ALL"

#python run.py --output_name vanilla --base_model_name gpt2-medium --mask_filling_model_name t5-large \
#--n_perturbation_list 5 --n_samples 312 --pct_words_masked 0.1 --span_length 2 \
#--dataset squad --dataset_key context --skip_baselines
