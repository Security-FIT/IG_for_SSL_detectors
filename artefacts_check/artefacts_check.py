#!/usr/bin/env python3
"""
Compute per-utterance artifact metrics (silences, VAD counts, RMSE stats) directly
from your MLDFDataset DataLoader.

Matches the measurements used in the reference script (duration, speech length,
leading/trailing silence, voiced/unvoiced frame counts and ratio, RMS in first 100 ms,
RMSE mean, RMSE amplitude), but works on in‑memory tensors coming from your dataset.

Example
-------
from datasets.MLDF import MLDFDataset
import torch
from artifact_metrics_mldf import check_artifacts

dataset = MLDFDataset(root="/path/to/mldf")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
rows = check_artifacts(dataloader, save_csv="mldf_artifacts.csv")
print(rows[0])
"""

from __future__ import annotations
import math
import csv
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional
import os
import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

import numpy as np
import librosa
import torchaudio
import torch
import pandas as pd

from tqdm import tqdm


@dataclass
class Metrics:
    # Timing
    duration_s: float
    speech_length_s: float
    leading_silence_s: float
    trailing_silence_s: float

    # VAD frame stats
    voiced_frames: int
    unvoiced_frames: int
    voiced_unvoiced_ratio: float

    # RMSE stats
    rms_first_100ms: float
    rmse_mean: float
    rmse_amplitude: float

    # Identifiers (filled from dataloader)
    name: str
    tool: str
    gender: str


def compute_metrics_for_waveform(
    wf: np.ndarray,
    sr: int = 16000,
    *,
    frame_duration_ms: int = 30,
    ta_vad_params: Optional[Dict[str, float]] = None,
    rmse_threshold: float = 0.05,
    silence_frame_window: int = 18,
    delta: float = 0.01,
) -> Dict[str, float]:
    """
    Compute artifact metrics for a single mono waveform (numpy array, float in [-1, 1]).

    This mirrors the logic of the reference implementation that uses 30 ms frames,
    WebRTC VAD, and RMSE-based gating.
    """
    if wf.ndim != 1:
        wf = wf.squeeze()
    wf = wf.astype(np.float32)

    duration = float(len(wf)) / float(sr)

    # Frame geometry (30 ms, non-overlapping), in *samples*
    hop = int(sr * frame_duration_ms / 1000)
    if hop <= 0:
        raise ValueError("Invalid hop length derived from frame_duration_ms.")

    # RMSE computed framewise (frame_length == hop, hop_length == hop)
    rmse = librosa.feature.rms(y=wf, frame_length=hop, hop_length=hop)[0]
    rmse_mean = float(np.mean(rmse)) if rmse.size else 0.0
    rmse_amplitude = float(np.ptp(rmse)) if rmse.size else 0.0

    # Use torchaudio VAD once on the whole utterance to find leading/trailing silence
    wf_t = torch.from_numpy(wf).view(1, -1)  # (1, T)
    params = dict(sample_rate=sr)
    if ta_vad_params:
        params.update(ta_vad_params)

    # Trim front
    trimmed_front = torchaudio.functional.vad(wf_t, **params)
    leading_samples = int(wf_t.shape[-1] - trimmed_front.shape[-1])
    leading_samples = max(0, min(leading_samples, wf_t.shape[-1]))

    # Trim back: run VAD on reversed waveform
    reversed_wf = torch.flip(wf_t, dims=[-1])
    trimmed_back = torchaudio.functional.vad(reversed_wf, **params)
    trailing_samples = int(wf_t.shape[-1] - trimmed_back.shape[-1])
    trailing_samples = max(0, min(trailing_samples, wf_t.shape[-1]))

    leading_silence = leading_samples / float(sr)
    trailing_silence = trailing_samples / float(sr)

    # Speech region indices
    speech_start = leading_samples
    speech_end = len(wf) - trailing_samples
    speech_start = max(0, min(speech_start, len(wf)))
    speech_end = max(0, min(speech_end, len(wf)))

    # Speech length is duration minus margins
    speech_length = max(0.0, duration - (leading_silence + trailing_silence))

    # Per-frame voiced/unvoiced counts inside speech region using RMSE as gate
    total_voiced = 0
    total_unvoiced = 0
    n_frames = len(rmse)
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(wf))
        if end - start < hop:
            continue
        in_region = (start >= speech_start) and (end <= speech_end)
        if not in_region:
            continue
        is_voiced = bool(rmse[i] > rmse_threshold)
        if is_voiced:
            total_voiced += 1
        else:
            total_unvoiced += 1

    vur = (
        (float(total_voiced) / float(total_unvoiced))
        if total_unvoiced > 0
        else float("inf") if total_voiced > 0 else 0.0
    )
    # First 100 ms RMSE average
    first_100ms_frames = int(0.1 / (frame_duration_ms / 1000.0))
    if n_frames >= first_100ms_frames and first_100ms_frames > 0:
        rms_100ms = float(np.mean(rmse[:first_100ms_frames]))
    else:
        rms_100ms = float(np.mean(rmse)) if n_frames > 0 else 0.0

    return {
        "duration_s": duration,
        "speech_length_s": speech_length,
        "leading_silence_s": float(leading_silence),
        "trailing_silence_s": float(trailing_silence),
        "voiced_frames": int(total_voiced),
        "unvoiced_frames": int(total_unvoiced),
        "voiced_unvoiced_ratio": float(vur),
        "rms_first_100ms": rms_100ms,
        "rmse_mean": rmse_mean,
        "rmse_amplitude": rmse_amplitude,
    }


def check_artifacts(
    dataloader: Iterable,
    *,
    sr: int = 16000,
    frame_duration_ms: int = 30,
    ta_vad_params: Optional[Dict[str, float]] = None,
    rmse_threshold: float = 0.05,
    silence_frame_window: int = 18,
    delta: float = 0.01,
    save_csv: Optional[str] = None,
) -> List[Dict[str, float]]:
    """
    Iterate your MLDF dataloader that yields (name, wf, tool, gender) and compute
    artifact metrics for each sample. Returns a list of dict rows and optionally
    writes a CSV/TSV depending on file extension.
    """
    rows: List[Dict[str, float]] = []

    for name, wf, tool, gender in tqdm(dataloader, desc="Artifacts"):
        # Expect batch_size == 1 as in the original snippet
        name = str(name[0]) if isinstance(name, (list, tuple)) or torch.is_tensor(name) else str(name)
        tool = str(tool[0]) if isinstance(tool, (list, tuple)) or torch.is_tensor(tool) else str(tool)
        gender = (
            str(gender[0]) if isinstance(gender, (list, tuple)) or torch.is_tensor(gender) else str(gender)
        )

        if torch.is_tensor(wf):
            wf_np = wf.squeeze().detach().cpu().numpy().astype(np.float32)
        else:
            wf_np = np.asarray(wf, dtype=np.float32).squeeze()

        metrics = compute_metrics_for_waveform(
            wf_np,
            sr=sr,
            frame_duration_ms=frame_duration_ms,
            ta_vad_params=ta_vad_params,
            rmse_threshold=rmse_threshold,
            silence_frame_window=silence_frame_window,
            delta=delta,
        )

        row = {
            "name": name,
            "tool": tool,
            "gender": gender,
            **metrics,
        }
        rows.append(row)

    if save_csv:
        delim = "\t" if save_csv.endswith(".tsv") else ","
        fieldnames = [
            "name",
            "tool",
            "gender",
            "duration_s",
            "speech_length_s",
            "leading_silence_s",
            "trailing_silence_s",
            "voiced_frames",
            "unvoiced_frames",
            "voiced_unvoiced_ratio",
            "rms_first_100ms",
            "rmse_mean",
            "rmse_amplitude",
        ]
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delim)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    return rows


class _MetricsDataset(Dataset):
    """Wrap an existing dataset and compute metrics inside worker processes.

    This lets you parallelize both I/O (dataset.__getitem__) and CPU compute
    by using DataLoader(num_workers>0).
    """

    def __init__(
        self,
        base_ds,
        *,
        sr: int = 16000,
        frame_duration_ms: int = 30,
        rmse_threshold: float = 0.05,
        silence_frame_window: int = 18,
        delta: float = 0.01,
        ta_vad_params: Optional[Dict[str, float]] = None,
    ):
        self.base_ds = base_ds
        self.sr = sr
        self.frame_duration_ms = frame_duration_ms
        self.rmse_threshold = rmse_threshold
        self.silence_frame_window = silence_frame_window
        self.delta = delta
        self.ta_vad_params = ta_vad_params

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        name, wf, tool, gender = self.base_ds[idx]
        name = str(name)
        tool = str(tool)
        gender = str(gender)
        if torch.is_tensor(wf):
            wf_np = wf.squeeze().detach().cpu().numpy().astype(np.float32)
        else:
            wf_np = np.asarray(wf, dtype=np.float32).squeeze()

        metrics = compute_metrics_for_waveform(
            wf_np,
            sr=self.sr,
            frame_duration_ms=self.frame_duration_ms,
            rmse_threshold=self.rmse_threshold,
            silence_frame_window=self.silence_frame_window,
            delta=self.delta,
            ta_vad_params=self.ta_vad_params,
        )
        return {
            "name": name,
            "tool": tool,
            "gender": gender,
            **metrics,
        }


def check_artifacts_parallel(
    dataset,
    *,
    num_workers: int = os.cpu_count() or 4,
    batch_size: int = 32,
    sr: int = 16000,
    frame_duration_ms: int = 30,
    rmse_threshold: float = 0.05,
    silence_frame_window: int = 18,
    delta: float = 0.01,
    ta_vad_params: Optional[Dict[str, float]] = None,
    save_csv: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Compute metrics in parallel using worker processes.

    Uses a wrapper Dataset so computation happens inside workers. Collects rows
    on the main process and optionally streams them to CSV.
    """
    # Avoid intra-op overthreading that can tank throughput in multi-proc mode
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    mds = _MetricsDataset(
        dataset,
        sr=sr,
        frame_duration_ms=frame_duration_ms,
        rmse_threshold=rmse_threshold,
        silence_frame_window=silence_frame_window,
        delta=delta,
        ta_vad_params=ta_vad_params,
    )

    dl = DataLoader(
        mds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=lambda batch: batch,  # keep as list of dicts
    )

    rows: List[Dict[str, float]] = []
    writer = None
    if save_csv:
        delim = "	" if save_csv.endswith(".tsv") else ","
        fieldnames = [
            "name",
            "tool",
            "gender",
            "duration_s",
            "speech_length_s",
            "leading_silence_s",
            "trailing_silence_s",
            "voiced_frames",
            "unvoiced_frames",
            "voiced_unvoiced_ratio",
            "rms_first_100ms",
            "rmse_mean",
            "rmse_amplitude",
        ]
        f = open(save_csv, "w", newline="")
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delim)
        writer.writeheader()

    try:
        for batch in tqdm(dl, desc=f"Artifacts x{num_workers}"):
            if writer is not None:
                for r in batch:
                    writer.writerow(r)
            rows.extend(batch)
    finally:
        if writer is not None:
            writer.writerow  # no-op to keep linter happy
            f.close()

    return rows


if __name__ == "__main__":
    import argparse
    
    # Simple dataset implementation for the IG recordings
    class IGDataset(Dataset):
        def __init__(self, csv_path, recordings_dir, sr=16000):
            self.df = pd.read_csv(csv_path)
            self.recordings_dir = recordings_dir
            self.sr = sr

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            file_id = str(row['FileID'])
            # Some file IDs might not have extensions in the CSV
            file_path = os.path.join(self.recordings_dir, f"{file_id}.flac")
            
            # Load and possibly resample
            wf, sr = torchaudio.load(file_path)
            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
                wf = resampler(wf)
            
            # Return tuple (name, wf, tool, gender)
            # Using 'Attack' as tool
            return file_id, wf, str(row['Attack']), str(row['Gender'])

    
    # Paths based on workspace structure
    CSV_PATH = "outputs/final_selection_132_details.csv"
    RECORDINGS_DIR = "recordings"
    OUTPUT_CSV = "outputs/ig_recordings_artefacts.csv"

    print(f"Loading files from {CSV_PATH} and audio from {RECORDINGS_DIR}...")
    ds = IGDataset(CSV_PATH, RECORDINGS_DIR)
    
    print("Running artifact checks...")
    # Using parallel processing
    check_artifacts_parallel(
        ds,
        num_workers=min(os.cpu_count() or 4, 4), # Cap workers to be safe
        batch_size=1, # Keep simple
        save_csv=OUTPUT_CSV,
        ta_vad_params=dict(trigger_level=7.0), # Default VAD params usually need tuning, but keeping generic or using higher threshold for cleaner speech
    )
    print(f"Done! Results saved to {OUTPUT_CSV}")
