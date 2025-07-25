#!/bin/bash
subj_ids=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
model_name="ENIGMA-all-11"
config_name="Alljoined-1.6M"

uv run train.py \
    --model_name "$model_name" \
    --subj_ids "${subj_ids[@]}" \
    --config_name "$config_name" \
    --batch_size 1024

uv run recon_inference.py \
    --model_name "$model_name" \
    --subj_ids "${subj_ids[@]}" \
    --config_name "$config_name"
    
uv run evaluate_recons.py \
    --model_name "$model_name" \
    --subj_ids "${subj_ids[@]}" \
    --config_name "$config_name"