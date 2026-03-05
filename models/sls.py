from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F

class SLS(nn.Module):
    """
    Sensitive Layer Selection
    Implemented by Qishan Zhang et al., 2024, based on the paper: https://dl.acm.org/doi/pdf/10.1145/3664647.3681345
    Code adapted from: https://github.com/QiShanZhang/SLSforASVspoof-2021-DF
    """
    def __init__(self, inputs_dim=768, outputs_dim=1):
        super(SLS, self).__init__()

        self.ins_dim = inputs_dim
        self.outs_dim = outputs_dim
        
        self.fixed_frames = 201
        hidden_dim = ceil((inputs_dim - 2) / 3) * ceil((self.fixed_frames - 2) / 3) 

        # Initialize layers
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(inputs_dim, outputs_dim)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(hidden_dim, inputs_dim)
        self.fc3 = nn.Linear(inputs_dim, outputs_dim)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def getAttenF(self, layerResult):
        # layerResult: [L, B, T, D] -> We need to iterate over L
        poollayerResult = []
        fullf = []
        
        # layerResult is [Nb_Layer, Batch, Frame_len, Dim] based on forward transpose
        for layer in layerResult:
            # layer: [B, T, D]
            layery = layer.transpose(1, 2) # [B, D, T]
            layery = F.adaptive_avg_pool1d(layery, 1) # [B, D, 1]
            layery = layery.transpose(1, 2) # [B, 1, D]
            poollayerResult.append(layery)

            x = layer.unsqueeze(1) # [B, 1, T, D]
            fullf.append(x)

        layery = torch.cat(poollayerResult, dim=1) # [B, L, D]
        fullfeature = torch.cat(fullf, dim=1)      # [B, L, T, D]
        return layery, fullfeature

    def forward(self, x):
        # Input x: [Nb_Layer, Batch, Frame_len, Dim]
        
        # 1. Pad/Cut to fixed frames (201) required by the Conv/Linear logic
        target_frames = self.fixed_frames
        if x.shape[2] < target_frames:
            pad_amt = target_frames - x.shape[2]
            x = F.pad(x, (0, 0, 0, pad_amt))
        elif x.shape[2] > target_frames:
            x = x[:, :, :target_frames, :]

        # 2. Compute H
        y0, fullfeature = self.getAttenF(x) # y0: [B, L, D], full: [B, L, T, D]

        # Upper branch
        y0 = self.fc0(y0) # [B, L, 1]
        y0 = self.sig(y0)
        y0 = y0.unsqueeze(-1) # [B, L, 1, 1] for broadcasting

        # Lower branch (Weighted Sum)
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1) # [B, T, D]
        fullfeature = fullfeature.unsqueeze(dim=1) # [B, 1, T, D] for Conv2d

        # Classifier part
        x = self.first_bn(fullfeature)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3)) # Reduces dimensions
        
        x = torch.flatten(x, 1) # Flatten
        x = self.fc1(x)         # Projection
        x = self.selu(x)

        x = self.fc3(x)
        # x = self.selu(x)
        # output = self.logsoftmax(x)

        return x
