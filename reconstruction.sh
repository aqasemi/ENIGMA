subj_ids=(8)
model_name="ENIGMA_sub-(${subj_ids[*]// /,})"
config_name="things_eeg2" # Which dataset config file to load

python train.py \
    --model_name "$model_name" \
    --subj_ids "${subj_ids[@]}" \
    --config_name "$config_name" 

python recon_inference.py \
    --model_name "$model_name" \
    --subj_ids "${subj_ids[@]}" \
    --config_name "$config_name"
    
python evaluate_recons.py \
    --model_name "$model_name" \
    --subj_ids "${subj_ids[@]}" \
    --config_name "$config_name"