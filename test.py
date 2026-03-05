#!/usr/bin/env python3

import torch
import os
import argparse
import soundfile as sf
import matplotlib.pyplot as plt

from models.wavlm_sls import WavLM_SLS
from models.wavlm_camhfa import WavLM_CAMHFA
from models.wavlm_aasist import WavLM_AASIST

from utils.audio_utils import load_audio
from utils.ig_utils import compute_ig_attributions, plot_ig_time_attr
from utils.ig_visualization import overlay_ig_on_waveform, overlay_ig_on_spectrogram

def main(args):
    audio_path = args.audio
    model_type = args.model.lower()
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n===== Running pipeline with model: {model_type.upper()} =====\n")

    # 1. Load Audio
    waveform, sr = load_audio(audio_path)
    
    # 2. Initialize Model
    if model_type == "sls":
        model = WavLM_SLS()
        processor = model.sls
    elif model_type == "camhfa": # Renamed from mhfa to camhfa to match new class
        model = WavLM_CAMHFA()
        processor = model.camhfa
    elif model_type == "aasist":
        model = WavLM_AASIST()
        processor = model.aasist
    else:
        raise ValueError("Model type must be: sls, camhfa, or aasist")

    # 3. Inference
    # We can use the model directly now
    logits = model(waveform)
    pred_prob = torch.sigmoid(logits).item()
    pred_class = 1 if pred_prob > 0.5 else 0
    print(f"Predicted class: {pred_class} (Prob: {pred_prob:.4f})")

    # 4. IG Attributions
    # Use the integrated method
    time_attr = model.get_attributions(waveform, target_class=0)
    time_attr_np = time_attr.abs().detach().cpu().numpy()
    # print(time_attr_np.shape, time_attr_np)

    # 5. Visualization
    
    overlay_ig_on_waveform(
        waveform, time_attr_np, sr, output_path=os.path.join(output_dir, f"{model_type}_IG_wf.png")
    )

    print(f"\nResults saved to: {output_dir}/\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, default="fake.flac")
    parser.add_argument("--model", type=str, default="sls", choices=["sls", "camhfa", "aasist"])
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()
    main(args)
