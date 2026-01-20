import os

from datasets import load_dataset
from huggingface_hub import snapshot_download

print("Connecting to HuggingFace...")
data_dir = snapshot_download(
    repo_id="gaia-benchmark/GAIA",
    repo_type="dataset",
    token=os.getenv("HF_TOKEN")
)

print(f"Dataset downloaded to: {data_dir}")
print("Loading test dataset...")

dataset = load_dataset(data_dir, "2023_level1", split="validation")
dataset = load_dataset(data_dir, "2023_level2", split="validation")
dataset = load_dataset(data_dir, "2023_level3", split="validation")
print(f"✓ Successfully loaded {len(dataset)} GAIA test examples")
print(f"✓ Dataset cached at: {data_dir}")
print("")
