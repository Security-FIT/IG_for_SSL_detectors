#!/usr/bin/env python3

import argparse
import os
import config

# Set Hugging Face cache directory before importing transformers (via models)
os.environ["HF_HOME"] = config.HF_HOME

import torch
import torch.nn as nn
import torch.distributed as dist
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np

from models.wavlm_sls import WavLM_SLS
from models.wavlm_camhfa import WavLM_CAMHFA
from models.wavlm_aasist import WavLM_AASIST
from utils.asvspoof5_dataset import get_asvspoof5_dataloader
from utils.metrics import calculate_EER, calculate_minDCF
from utils.ddp_utils import setup_ddp, cleanup_ddp, gather_eval_results

# Global variables for masking
bundle = None
aligner_model = None
labels = None
bonafide_silence_profile = None

def init_masking_globals(device):
    global bundle, aligner_model, labels, bonafide_silence_profile
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    aligner_model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    if os.path.exists("bona_fide_silence_profile.pt"):
        bonafide_silence_profile = torch.load("bona_fide_silence_profile.pt", map_location=device)


def evaluate(model, dataloader, device, save_scores_path=None, use_ddp=False, rank=0):
    model.eval()

    all_labels = []
    all_scores = []
    all_filenames = []

    if dist.is_initialized() and dist.get_rank() != 0:
        iterator = dataloader
    else:
        iterator = tqdm(dataloader, desc="Evaluating")

    with torch.no_grad():
        for filenames, waveforms, labels in iterator:
            waveforms = waveforms.to(device)

            logits = model(waveforms)
            scores = logits.cpu().numpy().flatten()

            # Ensure we store basic types (float, int, str) for pickling/gathering
            all_labels.extend(labels.numpy().tolist())
            all_scores.extend(scores.tolist())
            all_filenames.extend(filenames)

    # Gather results from all ranks (if DDP)
    all_labels, all_scores, all_filenames = gather_eval_results(all_labels, all_scores, all_filenames)

    eer = 0.0
    min_dcf = 0.0

    # Metrics and saving only on rank 0
    if rank == 0:
        all_labels_np = np.array(all_labels)
        all_scores_np = np.array(all_scores)

        if len(all_labels_np) > 0:
            eer = calculate_EER(all_labels_np, all_scores_np)
            min_dcf = calculate_minDCF(all_labels_np, all_scores_np)

            print(f"\nEvaluation Results:")
            print(f"EER: {eer:.4f} ({eer*100:.2f}%)")
            print(f"minDCF: {min_dcf:.4f}")

        if save_scores_path:
            print(f"Saving scores to {save_scores_path}")
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_scores_path), exist_ok=True)
            with open(save_scores_path, "w") as f:
                for filename, score, label in zip(all_filenames, all_scores, all_labels):
                    f.write(f"{filename} {score} {label}\n")

    if use_ddp:
        dist.barrier()

    return eer, min_dcf


def mask_noise(ssl_emb, waveform, sample_rate, device, bona_fide_silence_profile, alpha=0.9):
    """
    Masks the silence parts of SSL embeddings by interpolating them towards a global "bona fide silence" profile.
    This simulates adding natural background noise to the silence regions.
    
    Args:
        ssl_emb: The SSL embeddings to mask.
        waveform: The original audio waveform.
        sample_rate: The sample rate of the waveform.
        device: The device to run the aligner on.
        bona_fide_silence_profile: A pre-computed mean SSL embedding of bona fide silence.
        alpha: The interpolation factor (0.0 = no change, 1.0 = replace entirely with silence).
    """    
    # Resample if necessary
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, int(bundle.sample_rate))

    # Ensure waveform is 2D (batch, time) for the aligner
    if waveform.dim() == 3 and waveform.shape[1] == 1:
        aligner_waveform = waveform.squeeze(1)
    else:
        aligner_waveform = waveform

    # Get emissions (phoneme probabilities)
    with torch.inference_mode():
        emissions, _ = aligner_model(aligner_waveform.to(device))

    # Get the most likely phoneme for each frame
    phoneme_indices = torch.argmax(emissions, dim=-1)
    blank_idx = labels.index("-")

    # Find silence/blank frames
    silence_mask = phoneme_indices == blank_idx

    # ssl_emb is expected to be [Nb_Layer, Batch, Time, Dim]
    # We need to reshape/transpose potentially or handle dimensions carefully.
    
    nb_layers = ssl_emb.shape[0]
    batch_size = ssl_emb.shape[1]
    ssl_time = ssl_emb.shape[2]
    dim = ssl_emb.shape[3]
    
    # Ensure ssl_emb and mask have the same number of time frames
    num_frames = emissions.shape[1]
    
    if num_frames != ssl_time:
        silence_mask = (
            torch.nn.functional.interpolate(
                silence_mask.float().unsqueeze(1), size=ssl_time, mode="nearest"
            )
            .squeeze(1)
            .bool()
        )
        num_frames = ssl_time
        
    masked_ssl_emb = ssl_emb.clone()
    
    # Masking
    # ssl_emb: [Nb_Layer, Batch, Time, Dim]
    # silence_mask: [Batch, Time]
    # bona_fide_silence_profile: [Nb_Layer, Dim]
    
    bona_fide_tensor = bona_fide_silence_profile.to(ssl_emb.device) # [Nb_Layer, Dim]
    
    # Expand silence mask to match ssl_emb dimensions for masking
    # [Batch, Time] -> [1, Batch, Time, 1] -> [Nb_Layer, Batch, Time, Dim]
    mask_expanded = silence_mask.unsqueeze(0).unsqueeze(-1).expand(nb_layers, -1, -1, dim)
    
    # Expand bona_fide_profile to match ssl_emb dimensions
    # [Nb_Layer, Dim] -> [Nb_Layer, 1, 1, Dim] -> [Nb_Layer, Batch, Time, Dim]
    profile_expanded = bona_fide_tensor.unsqueeze(1).unsqueeze(1).expand(-1, batch_size, ssl_time, -1)
    
    # Apply masking using tensor operations instead of loops
    # masked = (1 - alpha) * original + alpha * profile
    masked_ssl_emb[mask_expanded] = (1 - alpha) * ssl_emb[mask_expanded] + alpha * profile_expanded[mask_expanded]
            
    return masked_ssl_emb


def mask_phonemes(ssl_emb, waveform, sample_rate, device, threshold_quantile=0.75):
    """
    Masks high-energy phonemes in SSL embeddings using a Wav2Vec2 aligner.
    Replaces masked frames with the mean of the nearest non-masked frames.
    """
    # Unpack dimensions
    nb_layers, batch_size, ssl_time, dim = ssl_emb.shape
    
    # Resample if necessary
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, int(bundle.sample_rate))

    # Ensure waveform is 2D (batch, time) for the aligner
    if waveform.dim() == 3 and waveform.shape[1] == 1:
        aligner_waveform = waveform.squeeze(1)
    else:
        aligner_waveform = waveform

    # Get emissions (phoneme probabilities)
    with torch.inference_mode():
        emissions, _ = aligner_model(aligner_waveform.to(device))

    # Calculate energy for each frame
    # The wav2vec2 model has a stride of 320 samples (20ms)
    stride = 320
    aligner_time = emissions.shape[1]
    
    # We need to compute energy mask for EACH item in the batch
    
    # Truncate waveform to multiple of stride
    w_len = aligner_waveform.shape[1]
    truncated_len = (w_len // stride) * stride
    # Note: aligner_time might be slightly different from w_len // stride due to convolutions
    
    masked_ssl_emb = ssl_emb.clone()
    
    for b_idx in range(batch_size):
        # Calculate frame energies for this sample
        
        # Frame extraction for energy
        current_waveform = aligner_waveform[b_idx] # [Time]
        
        # Just use list comprehension as before but per batch
        energies = []
        for i in range(aligner_time):
            start = i * stride
            end = min(start + stride, current_waveform.shape[0])
            frame = current_waveform[start:end]
            if frame.numel() > 0:
                energy = torch.sum(frame**2)
                energies.append(energy.item())
            else:
                energies.append(0.0)
        
        energies_tensor = torch.tensor(energies, device=device)
        
        threshold = torch.quantile(energies_tensor, threshold_quantile)
        high_energy_mask = energies_tensor > threshold # [Aligner_Time]
        
        # Interpolate mask to SSL time
        if aligner_time != ssl_time:
             high_energy_mask = (
                torch.nn.functional.interpolate(
                    high_energy_mask.float().view(1, 1, -1), size=ssl_time, mode="nearest"
                )
                .view(-1)
                .bool()
            )
        
        num_frames = ssl_time
        
        # Get indices to mask
        mask_indices = torch.where(high_energy_mask)[0]
        
        for i in mask_indices:
            # Find nearest non-masked frames
            # Simple linear search
            left_idx = i - 1
            while left_idx >= 0 and high_energy_mask[left_idx]:
                left_idx -= 1

            right_idx = i + 1
            while right_idx < num_frames and high_energy_mask[right_idx]:
                right_idx += 1
                
            # Compute replacement value
            # We want to replace ssl_emb[:, b_idx, i, :]
            
            replacement = None
            if left_idx >= 0 and right_idx < num_frames:
                replacement = (ssl_emb[:, b_idx, left_idx, :] + ssl_emb[:, b_idx, right_idx, :]) / 2
            elif left_idx >= 0:
                replacement = ssl_emb[:, b_idx, left_idx, :]
            elif right_idx < num_frames:
                replacement = ssl_emb[:, b_idx, right_idx, :]
            
            if replacement is not None:
                masked_ssl_emb[:, b_idx, i, :] = replacement
            # Else: all masked? leave as original (or zero) - original is fine or 0
            
    return masked_ssl_emb


def mask_word_boundaries(ssl_emb, waveform, sample_rate, device):
    """
    Masks word boundaries and transitions between words and silence in SSL embeddings
    using a Wav2Vec2 aligner. Replaces masked frames with the mean of the nearest non-masked frames.
    """

    # Resample if necessary
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, int(bundle.sample_rate))

    # Ensure waveform is 2D (batch, time) for the aligner
    if waveform.dim() == 3 and waveform.shape[1] == 1:
        aligner_waveform = waveform.squeeze(1)
    else:
        aligner_waveform = waveform

    # Get emissions (phoneme probabilities)
    with torch.inference_mode():
        emissions, _ = aligner_model(aligner_waveform.to(device))

    # Get the most likely phoneme for each frame
    phoneme_indices = torch.argmax(emissions, dim=-1)

    word_delimiter_idx = labels.index("|")
    blank_idx = labels.index("-")

    # Find word boundaries (where the predicted token is the word delimiter)
    word_boundary_mask = phoneme_indices == word_delimiter_idx

    # Find silence/blank frames
    blank_mask = phoneme_indices == blank_idx

    # Find transitions between words and silence
    # A transition is when we go from a non-blank token to a blank token, or vice versa
    shifted_left = torch.roll(blank_mask, shifts=-1, dims=-1)
    shifted_right = torch.roll(blank_mask, shifts=1, dims=-1)

    # A frame is a transition if it's not blank but its neighbour is blank
    transition_mask = (~blank_mask) & (shifted_left | shifted_right)

    # Combine word boundaries and transitions
    combined_mask = word_boundary_mask | transition_mask

    # Ensure ssl_emb and mask have the same number of frames
    # ssl_emb: [Nb_Layer, Batch, Time, Dim]
    num_frames = emissions.shape[1]
    
    nb_layers, batch_size, ssl_time, dim = ssl_emb.shape
    
    if num_frames != ssl_time:
        combined_mask = (
            torch.nn.functional.interpolate(
                combined_mask.float().unsqueeze(1), size=ssl_time, mode="nearest"
            )
            .squeeze(1)
            .bool()
        )
        num_frames = ssl_time

    masked_ssl_emb = ssl_emb.clone()

    # Mask word boundaries and transitions with mean of neighbouring frames
    # Iterate over batches and time frames.
    for b in range(batch_size):
        for i in range(num_frames):
            if combined_mask[b, i]:
                # Find nearest non-masked frames
                left_idx = i - 1
                while left_idx >= 0 and combined_mask[b, left_idx]:
                    left_idx -= 1

                right_idx = i + 1
                while right_idx < num_frames and combined_mask[b, right_idx]:
                    right_idx += 1

                if left_idx >= 0 and right_idx < num_frames:
                    # Apply to all layers
                    masked_ssl_emb[:, b, i, :] = (ssl_emb[:, b, left_idx, :] + ssl_emb[:, b, right_idx, :]) / 2
                elif left_idx >= 0:
                    masked_ssl_emb[:, b, i, :] = ssl_emb[:, b, left_idx, :]
                elif right_idx < num_frames:
                    masked_ssl_emb[:, b, i, :] = ssl_emb[:, b, right_idx, :]
                else:
                    masked_ssl_emb[:, b, i, :] = torch.zeros_like(ssl_emb[:, b, i, :])

    return masked_ssl_emb


def mask_spectral(waveform, sample_rate, device, band=(600, 1200), reduction_factor=0.1):
    """
    Masks higher frequencies in the spectrogram of the input waveform by lowering their energy.
    This avoids introducing artefacts that would be caused by hard nullification.
    """
    # Define STFT parameters
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    window = torch.hann_window(win_length).to(device)
    
    # Ensure waveform is 2D [Batch, Time] for stft call 
    # If it is [Batch, 1, Time], squeeze it
    original_dims = waveform.dim()
    if original_dims == 3 and waveform.shape[1] == 1:
        stft_input = waveform.squeeze(1)
    else:
        stft_input = waveform

    # Compute STFT
    # Input to stft must be 1D or 2D.
    stft = torch.stft(
        stft_input.to(device), 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        window=window, 
        return_complex=True
    )
    # stft is [Batch, Freq, Time]
    
    # Calculate frequencies for each bin
    freqs = torch.linspace(0, sample_rate / 2, stft.shape[1], device=device)
    
    # Find bins in the band
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    
    # Lower the energy of the bins in the band
    stft[:, band_mask, :] *= reduction_factor
    
    # Compute inverse STFT
    masked_waveform = torch.istft(
        stft, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        window=window, 
        length=waveform.shape[-1] # Use original time length
    )
    
    # Restore dimensions if we squeezed
    if original_dims == 3:
        masked_waveform = masked_waveform.unsqueeze(1)
    
    return masked_waveform


def mask_volume(waveform, device, reduction_factor=0.1):
    """
    Masks the entire waveform by lowering its volume (multiplying by a small factor).
    """
    return waveform * reduction_factor


def mask_compressor(waveform, sample_rate, device, threshold_db=-20, ratio=4.0, attack_ms=5.0, release_ms=50.0):
    """
    Applies dynamic range compression.
    1. Attenuates peaks above threshold_db.
    2. Applies makeup gain to bring the overall level back up, effectively boosting quiet parts.
    """
    waveform = waveform.to(device)
    
    # 1. Compute Amplitude (Envelope)
    # Using simple absolute value for peak detection
    amplitude = torch.abs(waveform) + 1e-8
    
    # Convert to dB
    db = 20 * torch.log10(amplitude)
    
    # 2. Calculate Gain Reduction (attenuation for loud parts)
    
    over_threshold = db > threshold_db
    gain_reduction_db = torch.zeros_like(db)
    
    # Calculate how much to reduce gain for peaks
    # We want to lower the gain, so this value should be negative
    gain_reduction_db[over_threshold] = -(1.0 - 1.0/ratio) * (db[over_threshold] - threshold_db)
    
    # 3. Makeup Gain
    makeup_gain_db = 10.0
    
    # Total gain to apply: Attenuation (negative) + Makeup (positive)
    total_gain_db = gain_reduction_db + makeup_gain_db
    
    # Convert to linear
    total_gain = torch.pow(10, total_gain_db / 20.0)
    
    # Apply to waveform
    compressed_waveform = waveform * total_gain
    
    # Hard clip to simulate the 'limit' of the pedal/circuit and avoid digital clipping
    compressed_waveform = torch.clamp(compressed_waveform, -1.0, 1.0)
    
    return compressed_waveform


def evaluate_masked(model, dataloader, device, mask_type, save_scores_path=None, use_ddp=False, rank=0):
    model.eval()

    all_labels = []
    all_scores = []
    all_filenames = []

    if dist.is_initialized() and dist.get_rank() != 0:
        iterator = dataloader
    else:
        iterator = tqdm(dataloader, desc=f"Evaluating with {mask_type} masking")

    with torch.no_grad():
        for filenames, waveforms, labels in iterator:
            waveforms = waveforms.to(device)

            # Apply time-domain masking if needed
            if mask_type == "spectral":
                waveforms = mask_spectral(waveforms, sample_rate=16000, device=device)
            elif mask_type == "volume":
                waveforms = mask_volume(waveforms, device=device)
            elif mask_type == "compressor":
                waveforms = mask_compressor(waveforms, sample_rate=16000, device=device)

            # Extract SSL features
            # Handle DDP wrapper if present
            if hasattr(model, 'module'):
                ssl_emb = model.module.extract_features(waveforms)
            else:
                ssl_emb = model.extract_features(waveforms)

            # Apply SSL-domain masking if needed
            if mask_type == "noise":
                ssl_emb = mask_noise(ssl_emb, waveforms, 16000, device, bonafide_silence_profile)
            elif mask_type == "phonemes":
                ssl_emb = mask_phonemes(ssl_emb, waveforms, 16000, device)
            elif mask_type == "word_boundaries":
                ssl_emb = mask_word_boundaries(ssl_emb, waveforms, 16000, device)

            # Forward pass through the rest of the model
            # Handle DDP wrapper if present
            base_model = model.module if hasattr(model, 'module') else model
            
            if hasattr(base_model, 'sls'):
                logits = base_model.sls(ssl_emb)
            elif hasattr(base_model, 'camhfa'):
                # CAMHFA expects [Batch, Dim, Time, Nb_Layer]
                # ssl_emb is [Nb_Layer, Batch, Time, Dim]
                logits = base_model.camhfa(ssl_emb.permute(1, 3, 2, 0))
            elif hasattr(base_model, 'aasist'):
                logits = base_model.aasist(ssl_emb)
            else:
                raise ValueError("Unknown model architecture for masked evaluation")

            scores = logits.cpu().numpy().flatten()

            # Ensure we store basic types (float, int, str) for pickling/gathering
            all_labels.extend(labels.numpy().tolist())
            all_scores.extend(scores.tolist())
            all_filenames.extend(filenames)

    # Gather results from all ranks (if DDP)
    all_labels, all_scores, all_filenames = gather_eval_results(all_labels, all_scores, all_filenames)

    eer = 0.0
    min_dcf = 0.0

    # Metrics and saving only on rank 0
    if rank == 0:
        all_labels_np = np.array(all_labels)
        all_scores_np = np.array(all_scores)

        if len(all_labels_np) > 0:
            eer = calculate_EER(all_labels_np, all_scores_np)
            min_dcf = calculate_minDCF(all_labels_np, all_scores_np)

            print(f"\nEvaluation Results ({mask_type} masking):")
            print(f"EER: {eer:.4f} ({eer*100:.2f}%)")
            print(f"minDCF: {min_dcf:.4f}")

        if save_scores_path:
            print(f"Saving scores to {save_scores_path}")
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_scores_path), exist_ok=True)
            with open(save_scores_path, "w") as f:
                for filename, score, label in zip(all_filenames, all_scores, all_labels):
                    f.write(f"{filename} {score} {label}\n")

    if use_ddp:
        dist.barrier()

    return eer, min_dcf


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASVspoof5 Models")
    parser.add_argument(
        "--model", type=str, required=True, choices=["sls", "camhfa", "aasist"], help="Model architecture"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    # Arguments now default to config values
    parser.add_argument(
        "--data_dir", type=str, default=config.DATA_DIR, help="Path to ASVspoof5 root directory"
    )
    parser.add_argument(
        "--eval_protocol", type=str, default=config.EVAL_PROTOCOL, help="Eval protocol filename"
    )

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device"
    )
    parser.add_argument("--save_scores", type=str, default=None, help="Path to save scores file (optional)")
    parser.add_argument(
        "--mask", type=str, default=None, choices=["noise", "phonemes", "word_boundaries", "spectral", "volume", "compressor"], 
        help="Apply masking strategy during evaluation"
    )

    args = parser.parse_args()

    # Setup DDP
    use_ddp, global_rank, local_rank = setup_ddp()

    if use_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
        print(f"Using device: {device}")

    if args.mask:
        init_masking_globals(device)

    # 1. Initialize Model
    print(f"Initializing model: {args.model}")
    # Note: freeze_wavlm doesn't matter for eval, but we pass True to avoid unnecessary gradient requirements if any
    if args.model == "sls":
        model = WavLM_SLS(freeze_wavlm=True)
    elif args.model == "camhfa":
        model = WavLM_CAMHFA(freeze_wavlm=True)
    elif args.model == "aasist":
        model = WavLM_AASIST(freeze_wavlm=True)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # 2. Load Checkpoint
    if global_rank == 0:
        print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    state_dict = None
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Handle DDP 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    model.to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 3. Data Loader
    print("Loading data...")
    eval_loader = get_asvspoof5_dataloader(
        root_dir=args.data_dir,
        protocol_file_name=args.eval_protocol,
        variant="eval",
        batch_size=args.batch_size,
        augment=False,
        distributed=use_ddp,
    )

    # 4. Evaluate
    if args.mask:
        evaluate_masked(model, eval_loader, device, args.mask, args.save_scores, use_ddp, global_rank)
    else:
        evaluate(model, eval_loader, device, args.save_scores, use_ddp, global_rank)

    cleanup_ddp()


if __name__ == "__main__":
    main()
