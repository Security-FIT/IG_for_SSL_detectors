import torch
import torch.nn as nn
from models.base_model import WavLMBaseModel
from models.mhfa import MHFA_Group_Conv2D
from utils.ig_utils import compute_ig_attributions

class WavLM_CAMHFA(WavLMBaseModel):
    def __init__(self, pretrained_path="microsoft/wavlm-base-plus", freeze_wavlm=True,
                 input_dim=768, compression_dim=128, 
                 head_nb=8, group_nb=64, nb_layer=13):
        super().__init__(pretrained_path, freeze_wavlm)
        
        # Renaming MHFA_Group_Conv2D to CAMHFA contextually here
        # Output dim set to 1 for BCE training (logit)
        self.camhfa = MHFA_Group_Conv2D(
            head_nb=head_nb,
            inputs_dim=input_dim,
            compression_dim=compression_dim,
            outputs_dim=1,
            group_nb=group_nb,
            nb_layer=nb_layer
        )

        # Precomputed mean features for baseline
        self.ig_baseline = torch.load("models/camhfa_mean.pt", map_location=torch.device('cpu'))

    def forward(self, x):
        # x: [Batch, Time]
        features = self.extract_features(x) # [Nb_Layer, Batch, Time, Dim]
        
        # MHFA_Group_Conv2D expects [Batch, Dim, Time, Nb_Layer]
        # Permute: 
        # Batch(1) -> 0
        # Dim(3) -> 1
        # Time(2) -> 2
        # Nb_Layer(0) -> 3
        features_permuted = features.permute(1, 3, 2, 0)
        
        logits = self.camhfa(features_permuted) # [Batch, 1]
        
        return logits

    def get_attributions(self, waveform, target_class=0):
        """
        Computes Integrated Gradients attributions for the input waveform.
        Returns time-domain attributions.
        """
        features = self.extract_features(waveform) # [Nb_Layer, Batch, Time, Dim]

        # Expand baseline to match features shape: [Nb_Layer, 1, 1, Dim] -> [Nb_Layer, Batch, Time, Dim]
        batch, time = features.shape[1], features.shape[2]
        expanded_baseline = self.ig_baseline.unsqueeze(1).unsqueeze(1).expand(-1, batch, time, -1)
        
        # MHFA_Group_Conv2D expects [Batch, Dim, Time, Nb_Layer]
        model_input = features.permute(1, 3, 2, 0)
        
        attributions = compute_ig_attributions(model_input, self.camhfa, None, target_class, expanded_baseline.permute(1, 3, 2, 0))
        
        # Input was [B, D, T, L]
        # Sum over Dim(1) and Layer(3) -> [B, T]
        time_attr = attributions.sum(dim=1).sum(dim=-1).squeeze() # [T]
        
        return time_attr
