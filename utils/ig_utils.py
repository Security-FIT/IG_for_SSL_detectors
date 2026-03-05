import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
import os

class ModelWrapper(nn.Module):
    """
    Wraps the feature_processor (SLS/MHFA/AASIST) and the classifier
    so IG can attribute to the inputs of the feature_processor.
    """
    def __init__(self, processor, classifier=None):
        super().__init__()
        self.processor = processor
        self.classifier = classifier

    def forward(self, x):
        # x shape: [Nb_Layer, Batch, Time, Dim] or similar
        out = self.processor(x)  # Returns [B, Output_Dim]
        if self.classifier is not None:
            out = self.classifier(out) # Returns [B, Num_Classes]
        return out

def compute_ig_attributions(inputs, processor, classifier, target_class, baseline):
    """
    inputs: Tensor [Nb_Layer, Batch, Time, Dim]
    """
    model = ModelWrapper(processor, classifier)
    model.eval()

    ig = IntegratedGradients(model)

    attributions = ig.attribute(
        inputs,
        baselines=baseline,
        target=target_class
    )
    return attributions
