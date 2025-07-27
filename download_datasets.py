from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="Alljoined/Alljoined-1.6M",
#     repo_type="dataset",
#     local_dir="/ibex/user/qasemiaa/enigma/data/alljoined",
#     allow_patterns=["preprocessed_eeg/*", "stimuli.zip"]
# )

# # unzip stimuli.zip
# import zipfile
# import os

# def unzip_stimuli(zip_path, extract_to):
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_to)
#     print(f"Extracted {zip_path} to {extract_to}")
#     return os.path.join(extract_to, "stimuli")

# unzip_stimuli("/ibex/user/qasemiaa/enigma/data/alljoined/stimuli.zip", "/ibex/user/qasemiaa/enigma/data/alljoined/stimuli")
# os.remove("/ibex/user/qasemiaa/enigma/data/alljoined/stimuli.zip")

snapshot_download(
    repo_id="LidongYang/EEG_Image_decode",
    repo_type="dataset",
    local_dir="/ibex/user/qasemiaa/enigma/data/eeg_things2",
    allow_patterns=["Preprocessed_data_250Hz/*", "emb_eeg/*", "fintune_ckpts/*", "ViT-H-14_features_train.pt", "ViT-H-14_features_test.pt"]
    # ignore_patterns=[""
)