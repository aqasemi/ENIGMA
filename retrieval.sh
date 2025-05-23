subj_ids=(8)
model_name="ENIGMA_retrieval_sub-(${subj_ids[*]// /,})"
config_name="things_eeg2" # Which dataset config file to load

python train.py \
    --model_name "$model_name" \
    --subj_ids "${subj_ids[@]}" \
    --config_name "$config_name" \
    --retrieval_only \
    --num_epochs 50

python recon_inference.py \
    --model_name "$model_name" \
    --subj_ids "${subj_ids[@]}" \
    --config_name "$config_name" \
    --retrieval_only