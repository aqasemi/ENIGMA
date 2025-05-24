# ENIGMA

An EEG-to-Image decoding model that reconstructs visual images from EEG brain signals using spatiotemporal CNNs and diffusion models.

## Installation

```bash
conda env create -f environment.yml
conda activate enigma
```

## Usage

### Image Reconstruction

Train and run the full reconstruction pipeline:

```bash
# Edit subject IDs and config in reconstruction.sh
./reconstruction.sh
```

Or run manually:
```bash
# Train model
python train.py --model_name ENIGMA_sub8 --subj_ids 8 --config_name things_eeg2

# Generate reconstructions  
python recon_inference.py --model_name ENIGMA_sub8 --subj_ids 8 --config_name things_eeg2

# Evaluate quality
python evaluate_recons.py --model_name ENIGMA_sub8 --subj_ids 8 --config_name things_eeg2
```

### Image Retrieval

Train and test retrieval-only model:

```bash
# Edit subject IDs and config in retrieval.sh
./retrieval.sh
```

Or run manually:
```bash
# Train retrieval model
python train.py --model_name ENIGMA_retrieval_sub8 --subj_ids 8 --config_name things_eeg2 --retrieval_only --num_epochs 50

# Test retrieval
python recon_inference.py --model_name ENIGMA_retrieval_sub8 --subj_ids 8 --config_name things_eeg2 --retrieval_only
```

## Configuration

Available dataset configs in `configs/`:
- `things_eeg2.json` - THINGS-EEG2 dataset (10 subjects, 63 channels)
- `Alljoined-1.6M.json` - Alljoined-1.6M dataset (20 subjects, 32 channels)  
- `THINGS-EEG2+Alljoined-1.6M.json` - Merged dataset (30 subjects, 32 channels)

## Output

- Model checkpoints/evaluation metrics: `train_logs/{model_name}/`
- Reconstructed images: `output/{model_name}/`

## License

See LICENSE file for details.