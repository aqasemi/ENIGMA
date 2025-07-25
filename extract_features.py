import os
import argparse
import json
from PIL import Image

import torch
import torch.nn.functional as F
import open_clip
import pandas as pd
from tqdm import tqdm

# --- Self-Contained CLIPEncoder Class (from source/models.py) ---
# This makes the script runnable without relying on the project's source package.
class CLIPEncoder:
    """A wrapper for the open_clip model to encode images and text."""
    def __init__(
        self,
        model_name="ViT-H-14",
        pretrained="laion2b_s32b_b79k",
        precision="fp32",
        batch_size: int = 64,
        device="cuda",
        cache_dir="cache",
        **kwargs,
    ):
        self.batch_size = batch_size
        self.device = device
        os.makedirs(cache_dir, exist_ok=True)
        
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            model_name, pretrained, precision, device=device, cache_dir=cache_dir, **kwargs
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print(f"CLIP model {model_name} loaded on {self.device}.")

    def encode_text(self, text, normalize=False):
        """Encodes a list of text strings into embeddings."""
        features = []
        for i in tqdm(
            range(0, len(text), self.batch_size), desc="CLIP Encoding text"
        ):
            batch_text = text[i : min(i + self.batch_size, len(text))]
            inputs = self.tokenizer(batch_text).to(self.device)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                batch_features = self.model.encode_text(inputs)
                if normalize:
                    batch_features = F.normalize(batch_features, dim=-1)
            features.append(batch_features.detach().cpu())
        features = torch.cat(features, dim=0)
        return features

    def encode_image(self, image_paths, normalize=False):
        """Encodes a list of image file paths into embeddings."""
        features = []
        for i in tqdm(
            range(0, len(image_paths), self.batch_size), desc="CLIP Encoding images"
        ):
            batch_paths = image_paths[i : min(i + self.batch_size, len(image_paths))]
            
            # Preprocess images in the batch
            batch_images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(self.preprocess(img))
                except Exception as e:
                    print(f"Warning: Could not process image {path}. Skipping. Error: {e}")
                    # Add a placeholder or handle as needed
                    continue
            
            if not batch_images:
                continue

            batch_images_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad(), torch.amp.autocast("cuda"):
                batch_features = self.model.encode_image(batch_images_tensor)
                if normalize:
                    batch_features = F.normalize(batch_features, dim=-1)

            features.append(batch_features.detach().cpu())
        print(features)
        features = torch.cat(features, dim=0)
        return features

def get_metadata_df(metadata_path_template, subjects, partition_key):
    """Loads and concatenates metadata for a list of subjects."""
    full_df = []
    print(f"Loading metadata for partition '{partition_key}'...")
    for subject in tqdm(subjects, desc="Loading subject metadata"):
        path = metadata_path_template.format(subject=subject)
        if not os.path.exists(path):
            print(f"Warning: Metadata file not found for {subject} at {path}. Skipping.")
            continue
        df = pd.read_parquet(path)
        full_df.append(df)
    
    if not full_df:
        raise FileNotFoundError("No metadata files could be loaded. Check your data paths.")

    stim_df = pd.concat(full_df)
    # Filter for the specific partition (e.g., 'stim_train' or 'stim_test')
    stim_df = stim_df[stim_df["partition"] == partition_key].reset_index(drop=True)

    return stim_df

def process_partition(df, clip_encoder):
    """Encodes all unique images and text labels in a dataframe."""
    
    # Get unique image paths and category names
    unique_image_paths = df["image_path"].unique().tolist()
    unique_category_names = df["category_name"].unique().tolist()

    # replace the stem in image paths by IMAGES_PATH
    images_path = "data/alljoined/stimuli/images"
    unique_image_paths = [os.path.join(images_path, os.path.basename(path)) for path in unique_image_paths]
    
    print(f"Found {len(unique_image_paths)} unique images and {len(unique_category_names)} unique text labels.")

    # Encode images and text
    image_embeddings = clip_encoder.encode_image(unique_image_paths)
    text_embeddings = clip_encoder.encode_text(unique_category_names)

    # Create feature dictionaries
    # The keys must match the values from the dataframe for lookup in EEGDataset
    image_features = {path: emb for path, emb in zip(unique_image_paths, image_embeddings)}
    text_features = {name: emb for name, emb in zip(unique_category_names, text_embeddings)}

    return {"image": image_features, "text": text_features}

def main(args):
    """Main function to drive feature extraction."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load dataset configuration
    config_file = os.path.join("configs", f"{args.config_name}.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, "r") as f:
        config = json.load(f)

    print(f"Loaded configuration for '{args.config_name}'.")

    # 2. Initialize CLIP Encoder
    clip_encoder = CLIPEncoder(device=device, batch_size=args.batch_size, cache_dir=args.cache_dir)
    
    # 3. Process Training Data
    print("\n--- Processing Training Partition ---")
    train_df = get_metadata_df(
        metadata_path_template=config["metadata_path"],
        subjects=config["subjects"],
        partition_key="stim_train"
    )
    features_train = process_partition(train_df, clip_encoder)

    # 4. Save Training Features
    train_features_path = config["features_train"]
    os.makedirs(os.path.dirname(train_features_path), exist_ok=True)
    torch.save(features_train, train_features_path)
    print(f"Successfully saved training features to: {train_features_path}")

    # 5. Process Test Data
    print("\n--- Processing Test Partition ---")
    test_df = get_metadata_df(
        metadata_path_template=config["metadata_path"],
        subjects=config["subjects"],
        partition_key="stim_test"
    )
    features_test = process_partition(test_df, clip_encoder)

    # 6. Save Test Features
    test_features_path = config["features_test"]
    os.makedirs(os.path.dirname(test_features_path), exist_ok=True)
    torch.save(features_test, test_features_path)
    print(f"Successfully saved test features to: {test_features_path}")

    print("\nFeature extraction complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CLIP features for the ENIGMA project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="things_eeg2",
        help="Name of the dataset config file to use (e.g., 'things_eeg2', 'Alljoined-1.6M')."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for the CLIP encoder."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory to cache downloaded models."
    )
    
    args = parser.parse_args()
    main(args)