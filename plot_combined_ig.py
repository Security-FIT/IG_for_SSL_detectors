import json
import numpy as np
import matplotlib.pyplot as plt
import os

fid = "E_0005076209"
models = ["aasist", "camhfa", "sls"]
colors = {"aasist": "red", "camhfa": "green", "sls": "blue"}

plt.figure(figsize=(10, 2.4))

# We only need the waveform from the first one
waveform_plotted = False
sample_rate = 16000

for model in models:
    json_path = f"outputs/IG/{fid}_{model}_diff_baseline.json"
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        continue
        
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if not waveform_plotted:
        waveform = np.array(data["waveform"])
        sample_rate = data["sample_rate"]
        T = len(waveform)
        time = np.linspace(0, T / sample_rate, num=T)
        plt.plot(time, waveform, label="Waveform", alpha=0.3, color="gray")
        waveform_plotted = True
        
    attr_raw = np.array(data["attributions_raw"])
    
    # Calculate time axis for attributions
    ig_time = np.linspace(0, len(attr_raw) * 320 / sample_rate, num=len(attr_raw))
    
    window_size = 6
    
    if len(attr_raw) > window_size: #and model != "camhfa":
        # Smooth RAW
        padded_attr = np.pad(attr_raw, (window_size // 2, window_size // 2), mode='edge')
        shape = padded_attr.shape[:-1] + (padded_attr.shape[-1] - window_size + 1, window_size)
        strides = padded_attr.strides + (padded_attr.strides[-1],)
        strided_attr = np.lib.stride_tricks.as_strided(padded_attr, shape=shape, strides=strides)
        median_trend = np.median(strided_attr, axis=1)
        median_trend = median_trend[:len(ig_time)]
        
        # Normalize
        norm_factor = np.max(np.abs(median_trend))
        if norm_factor == 0: norm_factor = 1.0
        
        attr_norm = median_trend / norm_factor
        
        plt.plot(ig_time, attr_norm, color=colors[model], alpha=0.7 if model != "sls" else 0.6, label=f"{model.upper() if model.upper() != 'CAMHFA' else 'CA-MHFA'} IG (smooth)")
    else:
        # Normalize if not smoothed
        norm_factor = np.max(np.abs(attr_raw))
        if norm_factor == 0: norm_factor = 1.0
        attr_norm = attr_raw / norm_factor

        plt.plot(ig_time, attr_norm, color=colors[model], alpha=0.7, label=f"{model.upper()} IG (raw)")

plt.axvspan(0.3, 0.75, color='red', alpha=0.2, label="AASIST primary cue")
plt.axvspan(2.31, 2.43, color="green", alpha=0.2, label="CA-MHFA primary cue")
plt.axvspan(3.4, 3.8, color="blue", alpha=0.15, label="SLS primary cue")

# plt.title(f"IG attributions for {fid} Across All Detectors")
plt.xlabel("Time (s)")
# plt.xlim(0, len(waveform) / sample_rate)
plt.xlim(0, 4)
plt.ylim(0, 1)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()

out_path = f"outputs/{fid}_combined_ig.pdf"
plt.savefig(out_path)
print(f"Saved plot to {out_path}")
