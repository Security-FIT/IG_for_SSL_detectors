import torch
import torch.nn as nn


class MHFA_Group_Conv2D(nn.Module):
    """
    Context-Aware Multi-head factorized attentive pooling layer.
    Implemented by Junyi Peng, 2025, based on the paper: https://ieeexplore.ieee.org/document/10889058
    Code adapted from: https://huggingface.co/JYP2024/CA-MHFA_WavlmBasePlus_VoxCeleb2/blob/main/wespeaker/models/ssl_backend/MHFA.py
    """

    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=1, group_nb=64, nb_layer=13):
        super(MHFA_Group_Conv2D, self).__init__()
        
        # Multi Q + Single K + Single V
        self.weights_k = nn.Parameter(data=torch.ones(nb_layer), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(nb_layer), requires_grad=True)

        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim
        self.group_nb = group_nb

        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        group_len = self.head_nb // self.group_nb
        self.att_head = nn.Conv2d(1, self.group_nb, (group_len+1, self.cmp_dim), bias=False, stride=1, padding=(group_len//2, 0))

        self.pooling_fc = nn.Linear(self.group_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]
        
        # Compute key and value
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Compression
        k = self.cmp_linear_k(k) # B, T, F
        v = self.cmp_linear_v(v) # B, T, F

        k_att = self.att_head(k.unsqueeze(1)) # B, Head, T, 1
        k_att = k_att.permute(0,2,1,3) 

        v = v.unsqueeze(-2) 

        # Attention pooling
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(k_att, dim=1)), dim=1)

        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        outs = self.pooling_fc(pooling_outs)
        return outs
