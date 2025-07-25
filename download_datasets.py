from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Alljoined/Alljoined-1.6M",
    repo_type="dataset",
    local_dir="data/alljoined",
    allow_patterns=["preprocessed_eeg/*", "stimuli.zip"]
)

# unzip stimuli.zip
import zipfile
import os

def unzip_stimuli(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")
    return os.path.join(extract_to, "stimuli")

unzip_stimuli("data/alljoined/stimuli.zip", "data/alljoined/stimuli")
os.remove("data/alljoined/stimuli.zip")