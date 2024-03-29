#!/bin/bash
#!/bin/bash
#python run.py --output_name E4_gpt3_quirky_child --batch_size 5 \
#--chatgpt --mask_filling_model_name t5-3b \
#--openai_model davinci \
#--n_perturbation_list 20 --n_samples 20 \
#--pct_words_masked 0.3 --span_length 2 \
#--dataset xsum --skip_baselines --prompt "Complete this in the voice of a quirky child who loves hot tea: "
#
python run.py --output_name E4_gpt3_quirky_child --batch_size 5 \
--chatgpt --mask_filling_model_name t5-3b \
--openai_model davinci \
--n_perturbation_list 20 --n_samples 20 \
--pct_words_masked 0.3 --span_length 2 \
--dataset writing --skip_baselines --prompt "Complete this in the voice of a quirky child who loves hot tea: "

exit


### EXPERIMENT 1 ###
# also try with writing
python run.py --output_name E1vanilla_xsum --scoring_model gpt2-medium \
--mask_filling_model_name t5-3b --base_model_name gpt2-xl \
--n_perturbation_list 1,2,5,10 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset xsum

python run.py --output_name E1PROPN_conc_xsum --scoring_model gpt2-medium \
--mask_filling_model_name t5-3b --base_model_name gpt2-xl \
--n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset xsum --skip_baselines --concentration "PROPN"

python run.py --output_name E1ADJ_conc_xsum --scoring_model gpt2-medium \
--mask_filling_model_name t5-3b --base_model_name gpt2-xl \
--n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset xsum --skip_baselines --concentration "ADJ"

python run.py --output_name E1NOUN_conc_xsum --scoring_model gpt2-medium \
--mask_filling_model_name t5-3b --base_model_name gpt2-xl \
--n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset xsum --skip_baselines --concentration "NOUN"

python run.py --output_name E1VERB_conc_xsum --scoring_model gpt2-medium \
--mask_filling_model_name t5-3b --base_model_name gpt2-xl \
--n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset xsum --skip_baselines --concentration "VERB"

python run.py --output_name E1ADV_conc_xsum --scoring_model gpt2-medium \
--mask_filling_model_name t5-3b --base_model_name gpt2-xl \
--n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset xsum --skip_baselines --concentration "ADV"

python run.py --output_name E1AVN_conc_xsum --scoring_model gpt2-medium \
--mask_filling_model_name t5-3b --base_model_name gpt2-xl \
--n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset xsum --skip_baselines --concentration "ALL"

python run.py --output_name E1STOP_conc_xsum --scoring_model gpt2-medium \
--mask_filling_model_name t5-3b --base_model_name gpt2-xl \
--n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset xsum --skip_baselines --concentration "STOP"

python run.py --output_name E1NONSTOP_conc_xsum --scoring_model gpt2-medium \
--mask_filling_model_name t5-3b --base_model_name gpt2-xl \
--n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset xsum --skip_baselines --concentration "NONSTOP"

python run.py --output_name E1FREQ_conc_xsum --scoring_model gpt2-medium \
--mask_filling_model_name t5-3b --base_model_name gpt2-xl \
--n_perturbation_list 1,2,5 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset xsum --skip_baselines --concentration "FREQ"



### EXPERIMENT 2 ####
# also try with xsum
python run.py --output_name E2baseline --scoring_model EleutherAI/gpt-j-6B \
--mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B\
--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset writing --skip_baselines \

python run.py --output_name E2novel1 --scoring_model EleutherAI/gpt-j-6B \
--mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B\
--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset writing --skip_baselines \
--prompt "I licked the cat. I drank from the moon. The ship sailed on the breadcrumbs. Surf me through the crackle of the night. "

python run.py --output_name E2gatsby --scoring_model EleutherAI/gpt-j-6B \
--mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B\
--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset writing --skip_baselines \
--prompt "So we beat on, boats against the current, borne back ceaselessly into the past. "

python run.py --output_name E2lolita --scoring_model EleutherAI/gpt-j-6B \
--mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B\
--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset writing --skip_baselines \
--prompt "Lolita, light of my life, fire of my loins. My sin, my soul. Lo-lee-ta: the tip of the tongue taking a trip of three steps down the palate to tap, at three, on the teeth. "

python run.py --output_name E2micemen --scoring_model EleutherAI/gpt-j-6B \
--mask_filling_model_name t5-3b --base_model_name EleutherAI/gpt-j-6B\
--n_perturbation_list 5,10,20 --n_samples 150 --pct_words_masked 0.3 --span_length 1 \
--dataset writing --skip_baselines \
--prompt "A stilted heron labored up into the air and pounded down river. For a moment the place was lifeless, and then two men emerged from the path and came into the opening by the green pool. "


### EXPERIMENT 3 ###
# todo

##### EXPERIMENT 4 #####
# will also run with xsum

#DONE ALREADY
#python run.py --output_name chatgpt --batch_size 5 \
#--chatgpt --mask_filling_model_name t5-3b \
#--scoring_model facebook/opt-2.7b \
#--n_perturbation_list 5,10,20,100 --n_samples 150 \
#--pct_words_masked 0.3 --span_length 2 \
#--dataset writing
#
#python run.py --output_name chatgpt --batch_size 5 \
#--chatgpt --mask_filling_model_name t5-3b \
#--scoring_model gpt2-xl \
#--n_perturbation_list 5,10,20,100 --n_samples 150 \
#--pct_words_masked 0.3 --span_length 2 \
#--dataset writing
#
#python run.py --output_name chatgpt --batch_size 5 \
#--chatgpt --mask_filling_model_name t5-3b \
#--scoring_model EleutherAI/gpt-neo-2.7B \
#--n_perturbation_list 5,10,20,100 --n_samples 150 \
#--pct_words_masked 0.3 --span_length 2 \
#--dataset writing
#
#python run.py --output_name chatgpt --batch_size 5 \
#--chatgpt --mask_filling_model_name t5-3b \
#--scoring_model EleutherAI/gpt-j-6B \
#--n_perturbation_list 5,10,20,100 --n_samples 150 \
#--pct_words_masked 0.3 --span_length 2 \
#--dataset writing

python run.py --output_name E4chatgpt_proust --batch_size 5 \
--chatgpt --mask_filling_model_name t5-3b \
--scoring_model EleutherAI/gpt-j-6B \
--n_perturbation_list 5,10,20,100 --n_samples 150 \
--pct_words_masked 0.3 --span_length 2 \
--dataset writing --skip_baselines --prompt "complete this in the style of a Marcel Proust: "

python run.py --output_name E4chatgpt_least_expected_way --batch_size 5 \
--chatgpt --mask_filling_model_name t5-3b \
--scoring_model EleutherAI/gpt-j-6B \
--n_perturbation_list 5,10,20,100 --n_samples 150 \
--pct_words_masked 0.3 --span_length 2 \
--dataset writing --skip_baselines --prompt "complete this in the least expected way: "

exit



## TESTING THE GPT3 STUFF
#python run.py --output_name openai --batch_size 5 \
#--openai_model davinci --mask_filling_model_name t5-large \
#--scoring_model gpt2-medium \
#--n_perturbation_list 5,10 --n_samples 150 \
#--pct_words_masked 0.3 --span_length 2 \
#--do_top_p --top_p 0.9 --mask_top_p 0.95 \
#--dataset writing
##--dataset squad --dataset_key context

