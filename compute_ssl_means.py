

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import os
import argparse
import config


os.environ["HF_HOME"] = config.HF_HOME


from models.wavlm_aasist import WavLM_AASIST
from models.wavlm_camhfa import WavLM_CAMHFA
from models.wavlm_sls import WavLM_SLS
from utils.asvspoof5_dataset import ASVspoof5Dataset, pad_collate_fn

def get_model(model_name, device):
    """Initializes model and loads best weights."""
    model_path = os.path.join("checkpoints", "wavlm-base-plus", f"{model_name}_best.pt")
    
    print(f"Loading {model_name} from {model_path}...")
    
    # Initialize model
    # We set freeze_wavlm=True because we only need inference
    if model_name == "aasist":
        model = WavLM_AASIST(freeze_wavlm=True)
    elif model_name == "camhfa":
        model = WavLM_CAMHFA(freeze_wavlm=True)
    elif model_name == "sls":
        model = WavLM_SLS(freeze_wavlm=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load weights
    try:
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found. Using default initialized weights (Pretrained WavLM + Random Backend).")
        else:
            state_dict = torch.load(model_path, map_location=device)
            # Handle potential DataParallel wrapping if keys have 'module.'
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            
            # Use strict=True first, if it fails because of missing keys (maybe wavlm wasn't saved?), try strict=False
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(f"Strict loading failed: {e}")
                print("Trying strict=False...")
                model.load_state_dict(state_dict, strict=False)
                
            print(f"Successfully loaded weights for {model_name}")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        
    model.to(device)
    model.eval()
    return model

def compute_mean(model, dataloader, device):
    running_sum = 0
    total_frames = 0
    
    with torch.no_grad():
        for _, waveform, _ in tqdm(dataloader, desc="Computing mean"):
            waveform = waveform.to(device)
            # waveform shape: [Batch, Length] assuming Batch Size 1 and squeezing happens in collate or not?
            # ASVspoof5Dataset returns (filename, waveform[1, T], label)
            # default collate stacks to [Batch, 1, T]
            
            if waveform.dim() == 3 and waveform.shape[1] == 1:
                waveform = waveform.squeeze(1) # [Batch, Time]
            
            # extract_features returns [Nb_Layer, Batch, Time, Dim]
            features = model.extract_features(waveform)
            
            # Sum over Batch(1) and Time(2) -> [Nb_Layer, Dim]
            batch_sum = features.sum(dim=(1, 2))
            
            # Number of frames contributed by this batch
            num_frames = features.shape[1] * features.shape[2]
            
            running_sum += batch_sum
                
            total_frames += num_frames
            
    if total_frames == 0:
        return None
        
    global_mean = running_sum / total_frames
    return global_mean

def main():
    parser = argparse.ArgumentParser(description="Compute global mean of SSL features.")
    parser.add_argument("--aasist", action="store_true", help="Process AASIST model")
    parser.add_argument("--camhfa", action="store_true", help="Process CAMHFA model")
    parser.add_argument("--sls", action="store_true", help="Process SLS model")
    args = parser.parse_args()

    models_to_process = []
    if args.aasist:
        models_to_process.append("aasist")
    if args.camhfa:
        models_to_process.append("camhfa")
    if args.sls:
        models_to_process.append("sls")
    
    if not models_to_process:
        print("No specific models requested. Processing ALL models.")
        models_to_process = ["aasist", "camhfa", "sls"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Collect all datasets
    datasets = []
    
    # Train
    if hasattr(config, "TRAIN_PROTOCOL") and config.TRAIN_PROTOCOL:
        try:
            print(f"Loading Train set: {config.TRAIN_PROTOCOL}")
            ds_train = ASVspoof5Dataset(config.DATA_DIR, config.TRAIN_PROTOCOL, variant="train")
            ds_train.protocol_df = ds_train.protocol_df[ds_train.protocol_df["KEY"] == "bonafide"].reset_index(drop=True)
            datasets.append(ds_train)
        except Exception as e:
            print(f"Skipping Train: {e}")

    # Dev - Skipped as per user request to use only training split
    # if hasattr(config, "DEV_PROTOCOL") and config.DEV_PROTOCOL:
    #     try:
    #         print(f"Loading Dev set: {config.DEV_PROTOCOL}")
    #         ds_dev = ASVspoof5Dataset(config.DATA_DIR, config.DEV_PROTOCOL, variant="dev")
    #         datasets.append(ds_dev)
    #     except Exception as e:
    #          print(f"Skipping Dev: {e}")

    # Eval - Skipped as per user request to use only training split
    # if hasattr(config, "EVAL_PROTOCOL") and config.EVAL_PROTOCOL:
    #      try:
    #         print(f"Loading Eval set: {config.EVAL_PROTOCOL}")
    #         ds_eval = ASVspoof5Dataset(config.DATA_DIR, config.EVAL_PROTOCOL, variant="eval")
    #         datasets.append(ds_eval)
    #      except Exception as e:
    #          print(f"Skipping Eval: {e}")
    
    if not datasets:
        print("No datasets successfully loaded.")
        return
    
    # Using batch_size=1 ensures we process raw length without padding zeros affecting the mean
    dataloader = DataLoader(datasets[0], batch_size=1, shuffle=False, num_workers=4, collate_fn=None)
    
    output_dir = "models/means"
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name in models_to_process:
        print(f"\nProcessing {model_name}...")
        try:
            model = get_model(model_name, device)
            
            mean = compute_mean(model, dataloader, device)
            
            if mean is not None:
                save_path = os.path.join(output_dir, f"{model_name}_mean.pt")
                torch.save(mean, save_path)
                print(f"Saved global mean (shape {mean.shape}) to {save_path}")
            else:
                print(f"Failed to compute mean for {model_name} (no frames?)")
                
            # Free memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to process {model_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
