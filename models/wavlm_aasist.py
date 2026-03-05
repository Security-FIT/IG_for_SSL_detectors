import torch
import torch.nn as nn
from models.base_model import WavLMBaseModel
from models.aasist import AASIST
from utils.ig_utils import compute_ig_attributions

class WavLM_AASIST(WavLMBaseModel):
    def __init__(self, pretrained_path="microsoft/wavlm-base-plus", freeze_wavlm=True, 
                 aasist_input_dim=768):
        super().__init__(pretrained_path, freeze_wavlm)
        
        # Output dim set to 1 for BCE training (logit)
        self.aasist = AASIST(inputs_dim=aasist_input_dim, outputs_dim=1)

        # Precomputed mean features for baseline
        self.ig_baseline = torch.load("models/aasist_mean.pt", map_location=torch.device('cpu'))

    def forward(self, x):
        # x: [Batch, Time]
        features = self.extract_features(x) # [Nb_Layer, Batch, Time, Dim]
        
        # AASIST expects [Nb_Layer, Batch, Time, Dim]
        logits = self.aasist(features) # [Batch, 1]
        
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

        attributions = compute_ig_attributions(features, self.aasist, None, target_class, expanded_baseline)
        
        # Collapse dimensions to get time-domain importance
        # Input: [Nb_Layer, Batch, Time, Dim] -> Sum over Layer (0) and Dim (3)
        time_attr = attributions.sum(dim=0).sum(dim=-1).squeeze() # [T]
        
        return time_attr
