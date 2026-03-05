# Dataset Configuration
# Update this path to point to your ASVspoof5 dataset root directory
DATA_DIR = "/path/to/your/dir" 

# Protocols
TRAIN_PROTOCOL = "ASVspoof5.train.tsv"
DEV_PROTOCOL = "ASVspoof5.dev.track_1.tsv"
EVAL_PROTOCOL = "ASVspoof5.eval.track_1.tsv"

# Training Configuration
OUTPUT_DIR = "checkpoints"

# Hugging Face Cache
# Stores downloaded models in the 'env' directory to keep them with the environment
HF_HOME = "/path/to/your/hf_cache"
