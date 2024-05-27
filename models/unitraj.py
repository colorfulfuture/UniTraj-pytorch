'''
Author: Yi Xu <xu.yi@northeastern.edu>
UniTraj Model
'''

import torch
import torch.nn as nn
from models.encoder_decoder import Past_Encoder, Future_Encoder, Decoder
from models.modules import Normal

class UniTraj(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.past_encoder = Past_Encoder(args)
        self.future_encoder = Future_Encoder(args)
        self.decoder = Decoder(args)
        # Posterior q(z|x,y)
        self.qz_layer = nn.Linear(2*args.model_dim, 2*args.z_dim)
        nn.init.normal_(self.qz_layer.weight, mean=0, std=0.01)
        nn.init.constant_(self.qz_layer.bias, 0)
        # Prior p(z|x)
        if args.learn_prior:
            self.pz_layer = nn.Linear(args.model_dim, 2*args.z_dim)
            nn.init.normal_(self.pz_layer.weight, mean=0, std=0.01)
            nn.init.constant_(self.pz_layer.bias, 0)

    def get_delta(self, m):
        _, T, _ = m.shape
        d_forw = torch.zeros_like(m).cuda()
        d_back = torch.zeros_like(m).cuda()
        m_flip = torch.flip(m, dims=[1]) # [B T N]
        for t in range(1, T):
            d_forw[:,t,:] = 1 + torch.sub(1, m[:,t,:])*d_forw[:,t-1,:]
            d_back[:,t,:] = 1 + torch.sub(1, m_flip[:,t,:])*d_back[:,t-1,:]
        d_forw = d_forw.unsqueeze(-1).repeat_interleave(self.args.delta_dim, dim=-1)
        d_back = d_back.unsqueeze(-1).repeat_interleave(self.args.delta_dim, dim=-1)
        return [d_forw, d_back]

    def set_input(self, x, m=None):
        vel = torch.zeros_like(x).cuda()
        if m is not None:
            x_m = torch.mul(x, m) # masked trajectory
            vel[:,1:,:,:] = x_m[:,1:,:,:] - x_m[:,:-1,:,:]
            vel_m = torch.mul(vel, m) # multipy with the mask as well
            input = torch.cat((x_m, vel_m, m[:,:,:,0:1]),dim=-1)
        else:
            m_cat = torch.ones_like(x).cuda()
            vel[:,1:,:,:] = x[:,1:,:,:] - x[:,:-1,:,:]
            input = torch.cat((x, vel, m_cat[:,:,:,0:1]),dim=-1)
        return input

    def add_category(self, x):
        B, T, N, _ = x.shape
        N_team = int((N-1)/2) # - ball
        category = torch.zeros(N, 3).cuda()
        category[0,0] = 1
        category[1:1+N_team,1] = 1
        category[1+N_team:N,2] = 1
        category = category.repeat(B,T,1,1)
        x = torch.cat((x, category), dim=-1)
        return x

    def forward(self, data, mask):
        '''
        Input: data: [B T N 2(xy)]
               mask: [B T N 2(xy)]
        Return: loss
        '''
        B, T, N, _ = data.shape
        y = data.clone() # GT for training and loss

        x_delta = self.get_delta(mask[:,:,:,0])
        assert len(x_delta) == 2

        x = self.add_category(self.set_input(data, mask))
        assert x.size()[-1] == self.args.input_dim
        # Transformer encoder
        x_feat = self.past_encoder(x, mask[:,:,:,0], x_delta)
        x_feat = x_feat.reshape(B, N, T, -1).permute(0, 2, 1, 3) # [B T N C]

        y_mask = torch.ones_like(y).cuda()
        y_in = self.add_category(self.set_input(y))
        assert y_in.size()[-1] == self.args.input_dim
        # Transformer encoder
        y_feat = self.future_encoder(y_in, y_mask[:,:,:,0])
        y_feat = y_feat.reshape(B, N, T, -1).permute(0, 2, 1, 3) # [B T N C]

        # Posterior q(z|x,y)
        z_latent = torch.cat((x_feat, y_feat), dim=-1)
        qz_param = self.qz_layer(z_latent) # [B T N 2Z]
        qz_dist = Normal(params = qz_param)
        qz_sample = qz_dist.reparameterize()

        # Prior p(z)
        if self.args.learn_prior:
            pz_param = self.pz_layer(x_feat)
            pz_dist = Normal(params=pz_param)
        else:
            pz_dist = Normal(mu=torch.zeros(B, T, N, self.args.z_dim).cuda(), 
                             logvar=torch.zeros(B, T, N, self.args.z_dim).cuda())

        # Decoder, use qz_sample
        y_hat = self.decoder(x_feat, qz_sample)

        kld_loss = self.calculate_kld_loss(qz_dist, pz_dist, B, T, N)
        recon_loss = self.calculate_recon_loss(y_hat, y)

        # K = 20
        x_feat_repeat = x_feat.unsqueeze(3).repeat_interleave(self.args.k, dim=3) # [B T N K C]
        if self.args.learn_prior:
            pz_param_k = self.pz_layer(x_feat_repeat)
            pz_dist_k = Normal(params=pz_param_k)
        else:
            pz_dist_k = Normal(mu=torch.zeros(B, T, N, self.args.k, self.args.z_dim).cuda(), 
                             logvar=torch.zeros(B, T, N, self.args.k, self.args.z_dim).cuda())
        pz_sample_k = pz_dist_k.reparameterize() # [B T N K Z]

        y_hat_diverse = self.decoder(x_feat_repeat, pz_sample_k, self.args.k) # [B T N K 2]

        diverse_loss = self.calculate_diverse_loss(y_hat_diverse, y)

        return kld_loss, recon_loss, diverse_loss

    def inference(self, data, mask):
        '''
        Input: data: [B T N 2(xy)]
               mask: [B T N 2(xy)]
        Return: y_hat: [B T N 2(xy)]
        '''
        B, T, N, _ = data.shape

        x_delta = self.get_delta(mask[:,:,:,0])
        assert len(x_delta) == 2

        x = self.add_category(self.set_input(data, mask)) # [B T N 8 (x y vx vy mask 0 0 0)]
        assert x.size()[-1] == self.args.input_dim
        # Transformer encoder
        x_feat = self.past_encoder(x, mask[:,:,:,0], x_delta)
        x_feat = x_feat.reshape(B, N, T, -1).permute(0, 2, 1, 3) # [B T N C]

        # K = 20
        x_feat_repeat = x_feat.unsqueeze(3).repeat_interleave(self.args.k, dim=3) # [B T N K C]
        # Prior p(z)
        if self.args.learn_prior:
            pz_param_k = self.pz_layer(x_feat_repeat)
            pz_dist_k = Normal(params=pz_param_k)
        else:
            pz_dist_k = Normal(mu=torch.zeros(B, T, N, self.args.k, self.args.z_dim).cuda(), 
                             logvar=torch.zeros(B, T, N, self.args.k, self.args.z_dim).cuda())

        pz_sample_k = pz_dist_k.reparameterize() # [B T N K Z]
        # Decoder, use pz_sample in inference
        y_hat_diverse = self.decoder(x_feat_repeat, pz_sample_k, self.args.k) # [B T N K 2]

        return y_hat_diverse
    
    def calculate_kld_loss(self, q, p, B, T, N):
        epsilon = torch.finfo(torch.float).eps
        kld_element = 0.5 * (2 * torch.log(p.std + epsilon) - 2 * torch.log(q.std + epsilon) \
                        + (q.std.pow(2) + (q.mu - p.mu).pow(2)) / p.std.pow(2) \
                        - 1)
        return torch.sum(kld_element)/(B*T*N)
    
    def calculate_recon_loss(self, y_hat, y):
        '''
        y_hat/y: [B T N 2]
        for all masked and not masked locs
        '''
        squared_diff = (y_hat - y) ** 2
        return torch.mean(squared_diff)
    
    def calculate_diverse_loss(self, y_hat_diverse, y):
        '''
        y_hat: [B T N K 2]
        y: [B T N 2]
        '''
        k = y_hat_diverse.shape[-2]
        y_repeat = y.unsqueeze(-2).repeat_interleave(k, dim=-2)
        squared_diff = (y_hat_diverse - y_repeat) ** 2
        squared_diff_min = torch.mean(squared_diff, dim=-2)
        return torch.mean(squared_diff_min)