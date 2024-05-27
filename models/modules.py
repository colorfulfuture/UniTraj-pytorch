'''
Author: Yi Xu <xu.yi@northeastern.edu>
Modules for UniTraj
'''

import torch
import torch.nn as nn
from mamba_ssm import Mamba

class Attn_Block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.patch_embed = nn.Sequential(
            nn.Linear(args.input_dim, args.model_dim, args.bias),
            nn.LayerNorm(args.model_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.p_dropout)
            )
        num_patches = args.num_agent + 1 + 1
        self.attn_layer= nn.MultiheadAttention(
            embed_dim = args.model_dim,
            num_heads = args.num_heads,
            dropout = args.p_dropout,
            batch_first = True
            )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, args.model_dim))
        self.pos_drop = nn.Dropout(p=args.p_dropout)
        self.norm1 = nn.LayerNorm(args.model_dim)
        self.norm2 = nn.LayerNorm(args.model_dim)
        self.norm3 = nn.LayerNorm(args.model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(args.model_dim, args.model_dim, args.bias),
            nn.LeakyReLU(),
            nn.Linear(args.model_dim, args.model_dim, args.bias),
            nn.Dropout(args.p_dropout)
            )
        self.mask_embed = nn.Sequential(
            nn.Linear(args.model_dim, args.model_dim, args.bias),
            nn.LayerNorm(args.model_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.p_dropout)
            )

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def get_mask_headpatch(self, mask, c_dim):
        mask_emb = self.mask_embed(mask.unsqueeze(-1).repeat(1, 1, 1, c_dim)) # [B T N C]
        if self.args.operator == 'mean':
            mask_patch = torch.mean(mask_emb, dim=2, keepdim=True) # [B T 1 C]
        elif self.args.operator == 'sum':
            mask_patch = torch.sum(mask_emb, dim=2, keepdim=True)
        elif self.args.operator == 'max':
            mask_patch, _ = torch.max(mask_emb, dim=2, keepdim=True)
        else:
            raise ValueError
        return mask_patch

    def forward(self, x, mask):
        '''
        Input: x [B T N input_dim]
               mask [B T N]
        '''
        B, T, N, _ = x.shape
        x = self.patch_embed(x) # [B T N C]

        # Generate mask head patch and cat
        C_emb = x.size()[-1]
        x_head = self.get_mask_headpatch(mask, C_emb)
        x = torch.cat((x_head, x), dim=2).reshape(B*T, N+1, -1) # [B*T N+1 C]

        x = x + self.pos_embed
        x = self.pos_drop(x)
        qkv = self.norm1(x)
        x_attn, _ = self.attn_layer(qkv, qkv, qkv)
        x = x_attn # Try no Res
        x = self.ffn(self.norm2(x)) + x 
        y = self.norm3(x)[:,1:,:].reshape(B, T, N, -1)
        return y

class Mamba_Block(nn.Module):
    def __init__(self, d_model, d_state, 
                 expand, conv, depth=5):
        super().__init__()
        self.block = nn.ModuleList(
            [Mamba(d_model,
                   d_state,
                   expand,
                   conv) 
                   for _ in range(depth)]
        )

    def forward(self, x):
        for blk in self.block:
            x = blk(x) + x
        return x

class Temporal_Decay(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Linear(in_channels, out_channels, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        Input: x: delta [B T N 2] 
        Return: [B*N T C]
        [B T N 2] --> [B T N C] --> [B*N T C]
        '''

        B, T, N, _ = x.size()
        x = self.relu(self.mlp(x)).permute(0, 2, 1, 3).reshape(B*N, T, -1)
        return torch.exp(-x)

class Normal:
    def __init__(self, mu=None, logvar=None, params=None):
        super().__init__()
        if params is not None:
            self.mu, self.logvar = torch.chunk(params, chunks=2, dim=-1)
        else:
            assert mu is not None
            assert logvar is not None
            self.mu = mu
            self.logvar = logvar
        self.std = torch.exp(0.5 * self.logvar)

    def reparameterize(self):
        eps = torch.randn_like(self.std)
        return eps * self.std + self.mu

    def kld(self, p=None):
        """ compute KL(q||p) """
        if p is None:
            kl = -0.5 * (1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        else:
            epsilon = torch.finfo(torch.float).eps
            term1 = (self.mu - p.mu) / (p.std + epsilon)
            term2 = self.std / (p.std + epsilon)
            kl = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        return kl