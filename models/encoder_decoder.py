'''
Author: Yi Xu <xu.yi@northeastern.edu>
Encoder and Decoder for UniTraj
'''

import torch
import torch.nn as nn
from models.modules import Attn_Block, Mamba_Block, Temporal_Decay

class Past_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.spatial_encoder = Spatial_Encoder(args)
        self.temporal_encoder = Temporal_Encoder(args)
        self.temporal_decay = Temporal_Decay(args.delta_dim, args.model_dim)

    def forward(self, x, mask, x_delta):
        '''
        Input: x: [B T N C_in]
               mask: [B T N]
               x_delta: list of 2 [B T N 2]
        Return: [B*N T C_out]
        '''
        x_spa = self.spatial_encoder(x, mask)
        gamma = [self.temporal_decay(delta) for delta in x_delta]
        x_ssm = self.temporal_encoder(x_spa, gamma)
        return x_ssm


class Future_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.spatial_encoder = Spatial_Encoder(args)
        self.temporal_encoder = Temporal_Encoder(args)

    def forward(self, x, mask):
        '''
        Input: x: [B T N C_in]
               mask: [B T N]
        Return: [B*N T C_out]
        '''
        x_spa = self.spatial_encoder(x, mask)
        x_ssm = self.temporal_encoder(x_spa)
        return x_ssm

class Positional_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
    
    def build_pos_enc(self):
        pass

    def forward(self, x):
        pass

class Spatial_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.attn_block = Attn_Block(args)

    def forward(self, x, mask=None):
        '''
        Input:
                x: [B T N C_in]
                mask: [B T N]
        Output:
                y: [B T N C_out]
        '''
        B, T, N, _ = x.shape
        x_attn = self.attn_block(x, mask) # [B T N C]
        return x_attn

class Temporal_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mlp = nn.Sequential(
            nn.Linear(args.model_dim, args.model_dim),
            nn.LayerNorm(args.model_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.p_dropout)
            )
        self.ssm_forw = Mamba_Block(
            args.model_dim,
            args.state_dim,
            args.expand,
            args.conv_dim,
            args.tem_depth
            )
        self.ssm_back = Mamba_Block(
            args.model_dim,
            args.state_dim,
            args.expand,
            args.conv_dim,
            args.tem_depth
            )

    def forward(self, x, gamma=None):
        '''
        Input: x: [B T N C_in]
               gamma: [list 2] for past feature [B*N T C_in]
        Return: y: [B N L C_out]
        '''
        B, T, N, _ = x.shape
        # Forward
        x_forw = self.mlp(x).permute(0, 2, 1, 3).reshape(B*N, T, -1)

        # Backward
        _x_flip = torch.flip(x, dims=[1])
        x_back = self.mlp(_x_flip).permute(0, 2, 1, 3).reshape(B*N, T, -1)

        # Element-wise multiply with delta
        if gamma is not None:
            x_forw = x_forw * gamma[0]
            x_back = x_back * gamma[1]

        # Mamba block
        x_ssm_forw = self.ssm_forw(x_forw)
        x_ssm_back = self.ssm_back(x_back)

        y = (x_ssm_forw + torch.flip(x_ssm_back, dims=[1])) # [B*N L C]
        return y

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.out_layer = nn.Sequential(
            nn.Linear(args.model_dim + args.z_dim, args.z_dim),
            nn.Linear(args.z_dim, args.output_dim)
        )

    def forward(self, x, z, k=None):
        '''
        Input: x [B T N C]/[B T N K Z]
               z [B T N Z]/[B T N K Z]
        Return: y [B T N out_dim]
        '''
        hidden = torch.cat((x, z), dim=-1) # [B T N C+Z]
        y = self.out_layer(hidden) # [B T N 2]/[B T N K 2]
        return y