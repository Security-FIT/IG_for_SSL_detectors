#!/usr/bin/env python3
import glob
import json
import numpy as np
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import librosa
import os
from tqdm import tqdm
import warnings

# Suppress warnings that might clutter output
warnings.filterwarnings("ignore")

def compute_metrics_overtime(wf, target_len, sr=16000):
    """
    Computes RMSE and Voiceness curves from the waveform, 
    resampled to match the target_len (length of IG attributions).
    """
    # Ensure wf is numpy array
    wf = np.array(wf).astype(np.float32)
    
    # 1. Compute RMSE
    # Use a relatively fine hop length to capture details before downsampling
    hop_length = 160 # 10ms at 16k
    n_fft = 512
    
    # Check if waveform is long enough
    if len(wf) < n_fft:
        # edge case: very short audio
        rmse = np.zeros(1)
        voiced = np.zeros(1)
    else:
        rmse = librosa.feature.rms(y=wf, frame_length=n_fft, hop_length=hop_length, center=True)[0]
        
        # 2. Compute Voiceness (using RMSE threshold as proxy, matching artefacts_check.py)
        # artefacts_check.py uses 0.05 threshold.
        rmse_threshold = 0.05
        voiced = (rmse > rmse_threshold).astype(float)

    # 3. Resample to target_len
    # We use linear interpolation
    if len(rmse) == 0:
         return np.zeros(target_len), np.zeros(target_len)

    original_indices = np.linspace(0, 1, num=len(rmse))
    target_indices = np.linspace(0, 1, num=target_len)
    
    rmse_resampled = np.interp(target_indices, original_indices, rmse)
    voiced_resampled = np.interp(target_indices, original_indices, voiced)
    
    return rmse_resampled, voiced_resampled

def process_file(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if 'waveform' not in data or 'attributions' not in data:
            return None
            
        wf = data['waveform']
        attributions = np.array(data['attributions'])
        sr = data.get('sample_rate', 16000)
        
        target_len = len(attributions)
        if target_len < 2:
            return None # Cannot correlate
            
        rmse_curve, voiced_curve = compute_metrics_overtime(wf, target_len, sr)
        
        # Compute correlations
        # We correlate the metric curve with the attribution curve
        
        # Pearson
        p_rmse, _ = pearsonr(rmse_curve, attributions)
        p_voiced, _ = pearsonr(voiced_curve, attributions)
        
        # Spearman
        s_rmse, _ = spearmanr(rmse_curve, attributions)
        s_voiced, _ = spearmanr(voiced_curve, attributions)
        
        # Safe handling of NaNs (e.g. constant signal)
        def clean(val):
            return val if not np.isnan(val) else 0.0
            
        return {
            "filename": os.path.basename(filepath),
            "pearson_rmse": clean(p_rmse),
            "pearson_voiced": clean(p_voiced),
            "spearman_rmse": clean(s_rmse),
            "spearman_voiced": clean(s_voiced),
            "length_samples": len(wf),
            "length_frames": target_len
        }
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    output_dir = "outputs"
    ig_dir = os.path.join(output_dir, "IG")
    
    # Pattern for diff_baseline files
    display_pattern = os.path.join(ig_dir, "*_diff_baseline.json")
    files = glob.glob(display_pattern)
    
    print(f"Found {len(files)} files matching {display_pattern}")
    
    results = []
    for fp in tqdm(files, desc="Processing IG files"):
        res = process_file(fp)
        if res:
            # Parse filename to get details (E_0000020884_aasist_diff_baseline.json)
            # We want ID and Model ideally
            fname = res['filename']
            parts = fname.replace('_diff_baseline.json', '').split('_')
            # Assuming format E_ID_MODEL
            # E_0000020884_aasist -> ID=E_0000020884, Model=aasist
            if len(parts) >= 3:
                res['file_id'] = parts[0] + "_" + parts[1]
                res['model'] = "_".join(parts[2:])
            else:
                 res['file_id'] = fname
                 res['model'] = "unknown"
                 
            results.append(res)
            
    df = pd.DataFrame(results)
    
    if not df.empty:
        out_path = os.path.join(output_dir, "ig_artefact_correlations.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved results to {out_path}")
        print("Summary of correlations (Mean):")
        print(df[["pearson_rmse", "pearson_voiced", "spearman_rmse", "spearman_voiced"]].mean())
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
