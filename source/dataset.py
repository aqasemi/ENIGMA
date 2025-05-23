import os
import sys
import json
from typing import NamedTuple
import importlib.resources as pkg_resources

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.data import Dataset
import pandas as pd


class DataLoaderItem(NamedTuple):
    eeg: torch.Tensor
    class_id: int
    subject: str
    category: str
    img_path: str
    txt_vector: torch.Tensor
    img_vector: torch.Tensor
    txt_vector_norm: torch.Tensor
    img_vector_norm: torch.Tensor


@dataclass
class EEGDatasetConfig:
    metadata_path: str
    eeg_trials_train: str
    eeg_trials_test: str
    features_train: str
    features_test: str
    channels: list[str]
    subjects: list[str]

    @staticmethod
    def load_config(config_name="dataset"):

        config_path = "configs"
        config_file_path = f"configs/{config_name}.json"
        with open(config_file_path, "r") as config_file:
            config = json.load(config_file)
            return EEGDatasetConfig(**config)


class EEGDataset(Dataset):
    def __init__(
        self,
        config_name="things_eeg2",
        subjects=["sub-01"],
        split="train",
        seed=42,
    ):
        self.config = EEGDatasetConfig.load_config(config_name)
        for subject in subjects:
            assert (
                subject in self.config.subjects
            ), f"Subject {subject} not found in config"
        self.config.subjects = subjects
        self.split = split
        self.seed = seed

        # Determine which EEG trials and features files to load
        if self.split == "test":
            self.filename_template = self.config.eeg_trials_test
            self.features_file = self.config.features_test
            self.df_partition = "stim_test"
        else:
            self.filename_template = self.config.eeg_trials_train
            self.features_file = self.config.features_train
            self.df_partition = "stim_train"

        # Load EEG data and features
        self.stim_df = self.get_metadata()
        self.features = self.get_features()
        self.eeg_data = self.get_eeg_data()

        if self.df_partition == "stim_test":
            self.average_eeg()
        print(
            f"Finished loading {self.split} dataset, {self.eeg_data.shape} EEG"
            f" samples, {len(self.stim_df)} stimuli,"
            f" {len(self.features['text'].keys())} text features,"
            f" {len(self.features['image'].keys())} image features"
        )

    def get_metadata(self):
        stim_df = pd.DataFrame()

        # Gather metadata across subjects
        for i, subject in enumerate(
            tqdm(
                self.config.subjects,
                desc="Loading metadata into memory",
                file=sys.stderr,
            )
        ):
            filename = self.config.metadata_path.format(subject=subject)
            data = pd.read_parquet(filename, engine="pyarrow")
            stim_df = pd.concat([stim_df, data])
        # Filter to training or test set
        stim_df = stim_df[stim_df["partition"] == self.df_partition]
        if "dropped" in stim_df.columns:
            stim_df = stim_df[stim_df["dropped"] == False]

        # At this point our stim_df should contain rows corresponding to the preprocessed EEG file, so we reset the index here
        stim_df = stim_df.reset_index(drop=True)
        self.stim_indices = stim_df.index.tolist()
        print(f"Finished loading {len(stim_df)} metadata rows")
        return stim_df

    def get_eeg_data(self):
        """
        Loads preprocessed EEG data person subject from a specified npy file.
        Selects channels also specified by config.
        """
        eeg_data = []
        for subject in tqdm(
            self.config.subjects,
            desc="Loading EEG data into memory",
            file=sys.stderr,
        ):
            filename = self.filename_template.format(subject=subject)
            data = np.load(filename, allow_pickle=True)

            # Check data structure
            if not isinstance(data, dict):
                data = data.item()  # Handle old-style saving

            x = torch.from_numpy(data["preprocessed_eeg_data"]).float()
            x = self.select_channels(x, self.config.channels, data["ch_names"])
            eeg_data.append(x)
        eeg_data = torch.vstack(eeg_data)
        # Filter EEG data to contain only trials that have not been filtered out of our dataframe
        eeg_data = eeg_data[self.stim_indices]
        print("Done loading EEG data of shape ", eeg_data.shape)
        return eeg_data

    def get_features(self):
        """
        Features are dictionies of image and text CLIP encodings.
        If this dataset has been run before, the feature will be saved
        in a designated location.
        """
        features = torch.load(self.features_file)

        return features

    # Return all images in the stimulus dataframe
    def get_images(self):
        """
        Load and return all images from the filepaths in the stimulus dataframe.

        Returns:
            torch.Tensor: A tensor containing all ground truth images
                        with shape (n_images, channels, height, width).
        """
        filepaths = self.stim_df["image_path"].tolist()

        # Load and transform all images
        images = [
            transforms.ToTensor()(Image.open(filepath).convert("RGB"))
            for filepath in filepaths
        ]

        # Stack images into a single tensor (n_images, channels, height, width)
        return torch.stack(images)

    def get_image_features(self):
        unique_keys = []
        for key in self.features["image"].keys():
            if key.split("/")[-1] not in [
                k.split("/")[-1] for k in unique_keys
            ]:
                unique_keys.append(key)
        filtered_image_features = {
            k: self.features["image"][k] for k in unique_keys
        }
        return torch.stack(list(filtered_image_features.values()))

    def average_eeg(self):
        # To produce optimal results, we average across trial repetitions in the test set
        # Average EEG data entries in the test set for the same image, and update stim_df accordingly
        grouped_eeg = []
        grouped_rows = []
        for i, (_, group) in enumerate(
            self.stim_df.groupby(["image_path", "subject"])
        ):
            indices = group.index.tolist()
            avg_eeg = self.eeg_data[indices].mean(dim=0)
            new_row = group.iloc[0]
            grouped_eeg.append(avg_eeg)
            grouped_rows.append(new_row)
        self.eeg_data = torch.stack(grouped_eeg)
        self.stim_df = pd.DataFrame(grouped_rows).reset_index(drop=True)

    def __getitem__(self, index) -> DataLoaderItem:
        metadata = self.stim_df.iloc[index]
        eeg = self.eeg_data[index]
        img_vector = self.features["image"][metadata.image_path]
        txt_vector = self.features["text"][metadata.category_name]
        img_vector_norm = img_vector / img_vector.norm(dim=-1, keepdim=True)
        txt_vector_norm = txt_vector / txt_vector.norm(dim=-1, keepdim=True)
        return DataLoaderItem(
            eeg=eeg,
            subject=f"sub-{metadata.subject:02d}",
            class_id=metadata.category_num,
            category=metadata.category_name,
            img_path=metadata.image_path,
            txt_vector=txt_vector,
            img_vector=img_vector,
            txt_vector_norm=txt_vector_norm,
            img_vector_norm=img_vector_norm,
        )

    def __len__(self):
        return len(self.stim_df)

    @staticmethod
    def select_channels(data, selected, table):
        """
        Leveraged in `self.get_eeg_data` to sub-select channels as
        specified in the config.
        """
        indices = [table.index(channel) for channel in selected]
        return data[:, indices, :]
