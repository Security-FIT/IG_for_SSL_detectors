import torch
import torch.nn as nn
from transformers import WavLMModel

class WavLMBaseModel(nn.Module):
    def __init__(self, pretrained_path="microsoft/wavlm-base-plus", freeze_wavlm=True):
        super().__init__()
        # Try to load from local cache first to support offline compute nodes
        try:
            self.wavlm = WavLMModel.from_pretrained(pretrained_path, output_hidden_states=True, local_files_only=True)
        except OSError:
            # Fallback to downloading if not found (e.g. on login node)
            self.wavlm = WavLMModel.from_pretrained(pretrained_path, output_hidden_states=True, local_files_only=False)
        
        if freeze_wavlm:
            self.wavlm.eval()
            for param in self.wavlm.parameters():
                param.requires_grad = False
        else:
            self.wavlm.train()

    def extract_features(self, x):
        """
        Args:
            x: Input waveform tensor [Batch, Time]
        Returns:
            stacked_features: [Nb_Layer, Batch, Time, Dim]
        """
        # WavLM expects input values, we assume x is already normalized/processed if needed
        # or we can add the feature extractor here if raw audio is passed.
        # For now, assuming x is the raw waveform tensor.
        
        # Ensure input is [Batch, Time]
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
            
        outputs = self.wavlm(x)
        hidden_states = outputs.hidden_states
        
        # Stack layers: [Nb_Layer, Batch, Time, Dim]
        stacked_features = torch.stack(hidden_states, dim=0)
        return stacked_features

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented by subclasses")
