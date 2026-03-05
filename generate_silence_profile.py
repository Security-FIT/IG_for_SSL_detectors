import os
import torch
import torchaudio
from tqdm import tqdm
import glob
import config

# Set Hugging Face cache directory before importing transformers
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

from models.base_model import WavLMBaseModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize WavLM
    print("Initializing WavLM...")
    # Use the local path to the model to avoid HF cache issues
    wavlm_model = WavLMBaseModel(pretrained_path="microsoft/wavlm-base-plus", freeze_wavlm=True).to(device)
    wavlm_model.eval()

    # 2. Initialize Wav2Vec2 aligner
    print("Initializing Wav2Vec2 aligner...")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    aligner_model = bundle.get_model().to(device)
    aligner_model.eval()
    labels = bundle.get_labels()
    blank_idx = labels.index("-")

    # 3. Get available bona fide recordings from the subset
    print("Finding available bona fide recordings...")
    
    # Read the eval protocol to find bona fide files
    protocol_file = "scores/ASVspoof5.eval.track_1.tsv"
    bonafide_files = set()
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9 and parts[8] == "bonafide":
                bonafide_files.add(parts[1])
                
    # Get available flac files in the recordings directory
    available_flacs = glob.glob("recordings/*.flac")
    
    # Filter for bona fide files that are available
    valid_flacs = []
    for flac_path in available_flacs:
        filename = os.path.basename(flac_path).replace('.flac', '')
        if filename in bonafide_files:
            valid_flacs.append(flac_path)
            
    print(f"Found {len(valid_flacs)} available bona fide recordings in the subset.")

    if not valid_flacs:
        print("No valid bona fide recordings found. Exiting.")
        return

    sum_silence_embs = None
    total_silence_frames = 0

    with torch.no_grad():
        for flac_path in tqdm(valid_flacs, desc="Extracting silence profiles"):
            # Load waveform
            waveform, sample_rate = torchaudio.load(flac_path)
            waveform = waveform.to(device) # [1, Time]
            
            # Extract WavLM features
            # ssl_emb: [Nb_Layer, 1, Time_SSL, Dim]
            ssl_emb = wavlm_model.extract_features(waveform.unsqueeze(0)) 
            
            # Extract aligner emissions
            aligner_waveform = waveform
            
            # Resample if necessary
            if sample_rate != bundle.sample_rate:
                aligner_waveform = torchaudio.functional.resample(aligner_waveform, sample_rate, int(bundle.sample_rate))
                
            emissions, _ = aligner_model(aligner_waveform) # [1, Time_Aligner, Num_Labels]
            phoneme_indices = torch.argmax(emissions, dim=-1) # [1, Time_Aligner]
            
            silence_mask = phoneme_indices == blank_idx # [1, Time_Aligner]
            
            # Interpolate silence mask to match SSL time dimension
            num_frames = emissions.shape[1]
            ssl_num_frames = ssl_emb.shape[2]
            
            if num_frames != ssl_num_frames:
                silence_mask = torch.nn.functional.interpolate(
                    silence_mask.float().unsqueeze(1), size=ssl_num_frames, mode="nearest"
                ).squeeze(1).bool()
            
            # silence_mask is now [1, Time_SSL]
            
            b_silence_mask = silence_mask[0] # [Time_SSL]
            b_ssl_emb = ssl_emb[:, 0, :, :] # [Nb_Layer, Time_SSL, Dim]
            
            # Get silence frames for this recording
            b_silence_embs = b_ssl_emb[:, b_silence_mask, :] # [Nb_Layer, Num_Silence_Frames, Dim]
            
            num_silence_frames = b_silence_embs.shape[1]
            if num_silence_frames > 0:
                if sum_silence_embs is None:
                    sum_silence_embs = b_silence_embs.sum(dim=1) # [Nb_Layer, Dim]
                else:
                    sum_silence_embs += b_silence_embs.sum(dim=1)
                total_silence_frames += num_silence_frames

    if total_silence_frames > 0:
        mean_silence_profile = sum_silence_embs / total_silence_frames
        print(f"\nExtracted profile from {len(valid_flacs)} recordings, {total_silence_frames} silence frames.")
        print(f"Profile shape: {mean_silence_profile.shape}")
        
        torch.save(mean_silence_profile, "bona_fide_silence_profile.pt")
        print("Saved to bona_fide_silence_profile.pt")
    else:
        print("\nNo silence frames found!")

if __name__ == "__main__":
    main()
