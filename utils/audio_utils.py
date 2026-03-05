import torch
import librosa
import torchaudio.transforms as T

def load_audio(path, target_sr=16000):
    waveform, sample_rate = librosa.load(path, sr=None)
    waveform = torch.tensor(waveform).unsqueeze(0)

    if waveform.shape[0] > 1: 
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sr:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform, target_sr

