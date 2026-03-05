import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import json


def overlay_ig_on_waveform(waveform, attributions, sample_rate, output_path, global_max=None):
    """
    Args:
        waveform: Tensor [1, T]
        attributions: numpy or tensor [T] or [Frames]
        sample_rate: int
        output_path: filename to save output
        global_max: float, optional global maximum for normalization
    """
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.detach().cpu().numpy()

    if isinstance(waveform, torch.Tensor):
        waveform_np = waveform.squeeze().cpu().numpy()
    else:
        waveform_np = waveform
    
    T = len(waveform_np)

    time = np.linspace(0, T / sample_rate, num=T)
    ig_time = np.linspace(0, len(attributions) * 320 / sample_rate , num=len(attributions))

    norm_factor = global_max if global_max is not None else np.max(np.abs(attributions))
    if norm_factor == 0:
        norm_factor = 1.0

    plt.figure(figsize=(15, 3))
    plt.plot(time, waveform_np, label="Waveform", alpha=0.5)
    plt.plot(ig_time, np.maximum(attributions / norm_factor, 0), color="red", alpha=0.7, label="IG Attribution (spoof evidence)")
    plt.plot(ig_time, -np.maximum(-attributions / norm_factor, 0), color="green", alpha=0.7, label="IG Attribution (bonafide evidence)")
    # Add median trend
    window_size = 12  # Adjust as needed, number of time frames (320 samples)
    if len(attributions) > window_size:
        # Pad at edges to keep length consistent
        padded_attr = np.pad(attributions, (window_size // 2, window_size // 2), mode='edge')
        # Create a sliding window view
        shape = padded_attr.shape[:-1] + (padded_attr.shape[-1] - window_size + 1, window_size)
        strides = padded_attr.strides + (padded_attr.strides[-1],)
        strided_attr = np.lib.stride_tricks.as_strided(padded_attr, shape=shape, strides=strides)
        median_trend = np.median(strided_attr, axis=1)
        # Ensure lengths match in case of odd/even window issues
        median_trend = median_trend[:len(ig_time)]
        # Normalize for plotting
        plt.plot(ig_time, median_trend / norm_factor, color="purple", alpha=1.0, label="Median Trend")

    # plt.title("Waveform + Integrated Gradients Attribution")
    plt.xlabel("Time (s)")
    plt.xlim(0, T / sample_rate)
    plt.ylim(-1, 1)
    plt.legend(bbox_to_anchor=(-0.05, 0.7), loc='upper right', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def overlay_ig_on_spectrogram(waveform, attributions, output_path, sample_rate=16000):
    """
    Overlay IG attributions on Mel-spectrogram.

    Args:
        waveform: Tensor [1, T]
        attributions: numpy or tensor [T] or [Frames]
        sample_rate: int
        output_path: filename to save output
    """
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.detach().cpu().numpy()

    waveform_np = waveform.squeeze().cpu().numpy()

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=waveform_np, sr=sample_rate, n_fft=1024,
        hop_length=512, n_mels=128, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize IG to match number of spectrogram frames
    num_frames = mel_db.shape[1]

    resized_attr = np.interp(
        np.arange(num_frames),
        np.linspace(0, num_frames, len(attributions)),
        attributions
    )

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(
        mel_db, sr=sample_rate, x_axis='time',
        y_axis='mel', cmap="viridis"
    )

    # Normalize IG for overlay height scale
    scaled_attr = resized_attr / resized_attr.max() * mel_db.shape[0]

    plt.plot(
        np.linspace(0, mel_db.shape[1] * 512 / sample_rate, num_frames),
        scaled_attr,
        color="red",
        linewidth=1.2,
        alpha=0.9,
        label="IG Attribution (normalized)"
    )

    plt.title("Mel Spectrogram + Integrated Gradients Attribution")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def save_ig_visualization_data(waveform, attributions, sample_rate, output_path, global_max=None):
    """
    Save visualization data to JSON for interactive plotting.
    
    Args:
        waveform: Tensor [1, T] or numpy array [T]
        attributions: numpy or tensor [T]
        sample_rate: int
        output_path: filename to save JSON output
        global_max: float, optional global maximum for normalization
    """
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.detach().cpu().numpy()
    
    if isinstance(waveform, torch.Tensor):
        waveform_np = waveform.squeeze().cpu().numpy()
    else:
        waveform_np = waveform

    T = len(waveform_np)

    time = np.linspace(0, T / sample_rate, num=T)
    ig_time = np.linspace(0, len(attributions) * 320 / sample_rate , num=len(attributions))

    # Calculate Median Trend
    median_trend = None
    window_size = 12  # Adjust as needed, number of time frames (320 samples)
    if len(attributions) > window_size:
        # Pad at edges to keep length consistent
        padded_attr = np.pad(attributions, (window_size // 2, window_size // 2), mode='edge')
        # Create a sliding window view
        shape = padded_attr.shape[:-1] + (padded_attr.shape[-1] - window_size + 1, window_size)
        strides = padded_attr.strides + (padded_attr.strides[-1],)
        strided_attr = np.lib.stride_tricks.as_strided(padded_attr, shape=shape, strides=strides)
        median_trend = np.median(strided_attr, axis=1)
        # Ensure lengths match in case of odd/even window issues
        median_trend = median_trend[:len(ig_time)]

    # Normalize for display
    # We save normalized values to make frontend rendering simpler
    norm_factor = global_max if global_max is not None else np.max(np.abs(attributions))
    if norm_factor == 0:
        norm_factor = 1.0

    attributions_norm = attributions / norm_factor
    if median_trend is not None:
        median_trend_norm = median_trend / norm_factor
    else:
        median_trend_norm = None

    data = {
        "sample_rate": sample_rate,
        "waveform": waveform_np.tolist(),
        "attributions": attributions_norm.tolist(),
        "median_trend": median_trend_norm.tolist() if median_trend_norm is not None else None,
        "attributions_raw": attributions.tolist(),
        "median_trend_raw": median_trend.tolist() if median_trend is not None else None,
        "normalization_factor": float(norm_factor), # Include used norm factor
        "is_global_norm": global_max is not None
    }

    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    # print(f"Saved visualization data to {output_path}")


if __name__ == "__main__":
    waveform = torch.randn(1, 16000)
    attributions = np.abs(np.sin(np.linspace(0, 10, 500)))  # lower resolution IG

    os.makedirs("outputs", exist_ok=True)

    overlay_ig_on_waveform(waveform, attributions, 16000, "outputs/ig_waveform_test.png")
    overlay_ig_on_spectrogram(waveform, attributions, 16000, "outputs/ig_spectrogram_test.png")
