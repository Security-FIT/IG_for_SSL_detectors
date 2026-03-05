
import os
import config

# Set Hugging Face cache directory
os.environ["HF_HOME"] = config.HF_HOME

from huggingface_hub import snapshot_download

print(f"Downloading WavLM model to {config.HF_HOME}...")
try:
    snapshot_download(repo_id="microsoft/wavlm-base-plus")
    print("Download complete.")
except Exception as e:
    print(f"Download failed: {e}")
