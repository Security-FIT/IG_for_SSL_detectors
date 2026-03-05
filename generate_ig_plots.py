#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm

# Import Models
from models.wavlm_sls import WavLM_SLS
from models.wavlm_camhfa import WavLM_CAMHFA
from models.wavlm_aasist import WavLM_AASIST

# Import Utils
from utils.audio_utils import load_audio
from utils.ig_visualization import overlay_ig_on_waveform, save_ig_visualization_data


def load_model(model_name, checkpoint_path, device="cuda"):
    print(f"Loading {model_name} from {checkpoint_path}...")

    if model_name == "aasist":
        model = WavLM_AASIST()
    elif model_name == "camhfa":
        model = WavLM_CAMHFA()
    elif model_name == "sls":
        model = WavLM_SLS()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load Weights
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Check if state_dict has 'model' key or is direct
        if "model" in state_dict:
            state_dict = state_dict["model"]

        # Handle DataParallel prefix 'module.'
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        try:
            model.load_state_dict(
                new_state_dict, strict=False
            )  # Strict=False to ignore WavLM keys if missing/different
            print("  Weights loaded successfully.")
        except Exception as e:
            print(f"  Warning loading weights: {e}")
    else:
        print(f"  Checkpoint not found at {checkpoint_path}!")

    model.to(device)
    model.eval()
    return model


def find_audio_file(fid, root_dir):
    # Try common locations
    # 1. Direct concatenation (if root is header)
    p1 = os.path.join(root_dir, fid + ".flac")
    if os.path.exists(p1):
        return p1

    # 2. flac/ subdirectory
    p2 = os.path.join(root_dir, "flac", fid + ".flac")
    if os.path.exists(p2):
        return p2

    return None


def main():
    parser = argparse.ArgumentParser(description="Generate IG plots for selected recordings.")
    parser.add_argument("--input_csv", required=True, help="Path to selection CSV")
    parser.add_argument("--audio_dir", required=True, help="Root directory of audio files")
    parser.add_argument("--output_dir", default="outputs/plots", help="Output directory for plots")
    parser.add_argument("--models_dir", default="models", help="Directory containing .pt checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # 1. Load Data
    print(f"Reading list from {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv)
    except:
        # Fallback for plain text
        with open(args.input_csv, "r") as f:
            ids = [line.strip() for line in f if line.strip()]
        df = pd.DataFrame({"FileID": ids})

    print(f"Found {len(df)} files to process.")

    # 2. Load Models
    models = {}
    checkpoints = {
        "aasist": os.path.join(args.models_dir, "aasist_best.pt"),
        "camhfa": os.path.join(args.models_dir, "camhfa_best.pt"),
        "sls": os.path.join(args.models_dir, "sls_best.pt"),
    }

    for m_name, ckpt in checkpoints.items():
        try:
            models[m_name] = load_model(m_name, ckpt, args.device)
        except Exception as e:
            print(f"Failed to load {m_name}: {e}")

    # 3. Process Files
    os.makedirs(args.output_dir, exist_ok=True)

    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
        fid = row["FileID"]

        # if i != 32:
        #     continue

        # Locate Audio
        audio_path = find_audio_file(fid, args.audio_dir)
        if not audio_path:
            print(f"Skipping {fid}: Audio file not found in {args.audio_dir}")
            continue

        try:
            # Load Audio
            waveform, sr = load_audio(audio_path)
            waveform = waveform.to(args.device)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # [1, T]

            # Generate Plot for each model
            for m_name, model in models.items():
                if model is None:
                    continue

                # Compute IG with respect to the "spoof" class (target_class=0)

                try:
                    time_attr = model.get_attributions(waveform, target_class=0)
                    time_attr_np = time_attr.detach().cpu().numpy()

                    # Determine label for plot title
                    label_str = row["Label"] if "Label" in row else "?"
                    attack_str = row["Attack"] if "Attack" in row else "?"

                    # Filename: FileID_Model.png
                    save_name = f"{fid}_{m_name}_diff_baseline.png"
                    save_path = os.path.join(args.output_dir, save_name)

                    title = f"{m_name.upper()} IG (Target: Spoof) | {fid} [{label_str}/{attack_str}]"

                    # Plot
                    overlay_ig_on_waveform(waveform.cpu(), time_attr_np, sr, output_path=save_path)
                    
                    # Save JSON for interactive plot
                    json_path = os.path.join(args.output_dir, f"{fid}_{m_name}_diff_baseline.json")
                    save_ig_visualization_data(waveform.cpu(), time_attr_np, sr, output_path=json_path)

                except Exception as e:
                    print(f"Error processing {fid} with {m_name}: {e}")

        except Exception as e:
            print(f"Error loading audio {fid}: {e}")

    print(f"Done. Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
