#!/bin/bash
subj_ids=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
model_name="ENIGMA-all-things"
config_name="Alljoined-1.6M"
# config_name="things_eeg2"

python -m torch.distributed.run --nproc_per_node=4 train.py \
    --model_name "$model_name" \
    --subj_ids "${subj_ids[@]}" \
    --config_name "$config_name" \
    --batch_size 1024 \
    --retrieval_txt_loss_scale 0.25

uv run recon_inference.py \
    --model_name "$model_name" \
    --subj_ids "${subj_ids[@]}" \
    --config_name "$config_name"
    
uv run evaluate_recons.py \
    --model_name "$model_name" \
    --subj_ids "${subj_ids[@]}" \
    --config_name "$config_name"